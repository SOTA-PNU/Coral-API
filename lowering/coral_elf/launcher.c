// Coral NPU ELF launcher — EmitC 경로.
//
// 바이트코드 경로와의 차이
// ------------------------
//   VMFB 없음          -> iree-c-embed-data 불필요, 100 MB 제한 무관
//   인터프리터 없음     -> module_create() 가 컴파일된 C 함수를 부른다
//   full HAL 필요       -> local-sync device + static library loader
//
// module_create() 는 IREE 의 module_impl_emitc.c 가 -DEMITC_IMPLEMENTATION
// 으로 우리 헤더를 include 해서 만들어낸다. 여기서 직접 include 하지 않는다.
//
// 결과는 전역 심볼로 노출한다. 베어메탈에는 printf 도 디버거도 없어서
// 시뮬레이터가 코어 정지 후 메모리를 직접 읽는 것이 유일한 관측 수단이다.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include <stddef.h>
#include "iree/vm/shims.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"

#include "model.h"            // <name>_library_query 선언 (커널 .o 와 짝)
#include "model_shapes.h"     // 생성됨: MODEL_* 매크로 + 입력 데이터

#if defined(MODEL_USE_BYTECODE)
// 바이트코드 판: objcopy 가 만든 심볼로 VMFB 블롭을 가리킨다.
// 인자 마샬링을 런타임이 자기 구조체로 직접 하므로 EmitC 의 32비트 ref
// 정렬 문제(iree_vm_abi_*_t 는 자연 정렬, EmitC 는 8정렬 가정)가 없다.
#include "iree/vm/bytecode/module.h"
extern const uint8_t _binary_model_vmfb_start[];
extern const uint8_t _binary_model_vmfb_end[];

static iree_status_t model_module_create(iree_vm_instance_t* instance,
                                         iree_allocator_t allocator,
                                         iree_vm_module_t** out_module) {
  iree_const_byte_span_t blob = iree_make_const_byte_span(
      _binary_model_vmfb_start,
      (iree_host_size_t)(_binary_model_vmfb_end - _binary_model_vmfb_start));
  // 블롭은 .ddr_data 에 상주하는 정적 데이터라 해제할 필요가 없다.
  return iree_vm_bytecode_module_create(
      instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE, blob,
      iree_allocator_null(), allocator, out_module);
}
#else
// EmitC 모듈 생성자. module_impl_emitc.c 가 정의한다.
extern iree_status_t module_create(iree_vm_instance_t* instance,
                                   iree_allocator_t allocator,
                                   iree_vm_module_t** out_module);
#define model_module_create module_create
#endif

// ---- 시뮬레이터가 읽어갈 전역 심볼 --------------------------------------
volatile int32_t inference_status = -1;      // 0 = 성공
volatile uint32_t inference_stage = 0;       // 어디까지 갔나 (1..9)
volatile uint32_t inference_status_code = 0;

// ---- 구간별 사이클 측정 (데모) ----------------------------------------
// 칩에는 printf 가 없다. 밖에서 보고 싶은 값은 고정 주소를 가진 전역에
// 써 두고, 호스트가 심볼 주소로 그 메모리를 읽어간다.
volatile uint32_t cyc_setup = 0;    // 런타임 준비에 쓴 사이클
volatile uint32_t cyc_invoke = 0;   // 실제 추론에 쓴 사이클

// RV32 는 64비트 사이클 카운터를 두 워드로 나눠 읽는다.
// 상위 워드를 두 번 확인하지 않으면 경계에서 값이 튄다.
static inline uint64_t rdcycle(void) {
  uint32_t hi, lo, hi2;
  do { __asm__ volatile("csrr %0, mcycleh" : "=r"(hi));
       __asm__ volatile("csrr %0, mcycle"  : "=r"(lo));
       __asm__ volatile("csrr %0, mcycleh" : "=r"(hi2)); } while (hi != hi2);
  return ((uint64_t)hi << 32) | lo;
}
// 모든 출력을 순서대로 이어 담는다. 크면 DDR(.ddr_bss) 로 간다.
float inference_output[MODEL_OUTPUT_TOTAL_ELEMS]
    MODEL_OUTPUT_SECTION __attribute__((aligned(64)));
// 출력별 합계. 호스트가 ref_output*.bin 으로 계산한 값과 비교하면
// 앞 32개만 덤프해도 전체 일치를 확인할 수 있다.
volatile float inference_out_sum[MODEL_OUTPUT_COUNT];
volatile uint32_t inference_out_count = 0;   // 실제로 회수한 출력 개수
volatile int32_t inference_argmax = -1;
volatile uint32_t device_step = 0;
#if defined(MODEL_USE_BYTECODE)
// 코어가 실제로 무엇을 보는지 확인한다. 호스트의 read_memory 는 DDR 을
// 못 읽는 듯하므로(칩 실측), 코어가 읽은 값을 DTCM 전역으로 옮겨 덤프한다.
volatile uint32_t vmfb_len = 0;
volatile uint32_t vmfb_w0 = 0;   // 기대값 0x000020a4
volatile uint32_t vmfb_w2 = 0;   // 기대값 'IREE' = 0x45455249
#endif
// EmitC shim 은 ref 슬롯을 8바이트 정렬로 배치한다(생성 코드의
// iree_host_align(x, 8)). 런타임 ABI 구조체는 자연 정렬이라 32비트에서
// iree_vm_ref_t 의 정렬이 4다. 둘이 어긋나면 ref 를 엉뚱한 오프셋에서 읽어
// "ref type mismatch" 가 난다. 실제 값을 칩에서 재서 확인한다.
volatile uint32_t abi_off_a1 = 0;      // 런타임이 첫 ref 를 읽는 오프셋
volatile uint32_t abi_sizeof = 0;
volatile uint32_t ref_sizeof = 0;
volatile uint32_t ref_align = 0;
// EmitC 모듈은 타입을 이름으로 조회해 테이블을 만든다 (model_emitc.h).
// 조회가 실패하면 이후 모든 ref 검사가 "ref type mismatch" 로 떨어진다.
volatile uint32_t type_fence = 0, type_bufview = 0, type_device = 0, type_buffer = 0;
// 실패한 status 의 메시지를 그대로 담아 시뮬레이터에서 덤프한다.
volatile char status_msg[512];
volatile uint32_t status_msg_len = 0;
volatile char msg_c0[32], msg_c1[32], msg_c2[32], msg_c3[32];

static void capture_status(iree_status_t st) {
  if (iree_status_is_ok(st) || status_msg_len) return;
  iree_host_size_t len = 0;
  if (iree_status_format(st, sizeof(status_msg) - 1, (char*)status_msg, &len)) {
    status_msg[len] = 0;
    status_msg_len = (uint32_t)len;
    for (int i = 0; i < 32; ++i) {
      msg_c0[i] = status_msg[i];
      msg_c1[i] = status_msg[32 + i];
      msg_c2[i] = status_msg[64 + i];
      msg_c3[i] = status_msg[96 + i];
    }
  } else {
    status_msg_len = 0xFFFFFFFFu;   // 메시지가 빌드에서 제거됨
  }
}   // create_device 내부 진행

// --- 메모리 진단 -----------------------------------------------------------
volatile uint32_t mem_ddr_ok = 0;      // DDR 직접 읽기/쓰기 성공?
volatile uint32_t mem_malloc_kb = 0;   // 성공한 최대 malloc (KB)
volatile uint32_t mem_heap_lo = 0, mem_heap_hi = 0;
volatile uint32_t mem_aligned_ok = 0;   // aligned_alloc 되는가
volatile uint32_t mem_aligned_addr = 0; // 반환 주소 (정렬 확인)
volatile uint32_t mem_memalign_ok = 0;  // newlib memalign 되는가
volatile uint32_t mem_memalign_addr = 0;
extern volatile uint32_t aa_called;     // 내 aligned_alloc 이 불렸나

extern char __heap_start__, __heap_end__;

static void probe_memory(void) {
  mem_heap_lo = (uint32_t)(uintptr_t)&__heap_start__;
  mem_heap_hi = (uint32_t)(uintptr_t)&__heap_end__;

  // DDR 직접 접근
  volatile uint32_t* ddr = (volatile uint32_t*)0x80000000u;
  ddr[0] = 0xC0FFEE01u; ddr[1024] = 0xC0FFEE02u;
  mem_ddr_ok = (ddr[0] == 0xC0FFEE01u && ddr[1024] == 0xC0FFEE02u) ? 1u : 0u;

  // newlib memalign 직접 테스트
  {
    extern void* memalign(size_t, size_t);
    void* m = memalign(64, 1024);
    mem_memalign_addr = (uint32_t)(uintptr_t)m;
    mem_memalign_ok = (m != NULL) ? 1u : 0u;
    if (m) free(m);
  }

  // aligned_alloc — IREE 가 디바이스 구조체 할당에 쓴다 (memory.c:310)
  void* a = aligned_alloc(64, 1024);
  mem_aligned_addr = (uint32_t)(uintptr_t)a;
  mem_aligned_ok = (a != NULL) ? 1u : 0u;
  if (a) free(a);

  // malloc 상한 (KB 단위로 위에서 아래로)
  for (uint32_t kb = 8192; kb >= 4; kb /= 2) {
    void* q = malloc((size_t)kb * 1024u);
    if (q) { free(q); mem_malloc_kb = kb; break; }
  }
}
volatile float inference_max_value = 0.0f;

static iree_status_t create_device(iree_allocator_t host_allocator,
                                   iree_hal_device_t** out_device) {
  iree_hal_sync_device_params_t params;
  iree_hal_sync_device_params_initialize(&params);

  // 컴파일된 커널은 이 query 함수로 노출된다.
  const iree_hal_executable_library_query_fn_t libraries[] = {
      MODEL_LIBRARY_QUERY,
  };
  iree_hal_executable_loader_t* loader = NULL;
  iree_status_t status = iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries,
      iree_hal_executable_import_provider_null(), host_allocator, &loader);
  if (iree_status_is_ok(status)) device_step = 1;

  iree_string_view_t identifier = iree_make_cstring_view("local-sync");
  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_create_heap(identifier, host_allocator,
                                            host_allocator, &device_allocator);
  }
  if (iree_status_is_ok(status)) device_step = 2;

  // local-sync 는 proactor_pool 이 NULL 이면 안 된다.
  // (sync_device.c 의 IREE_ASSERT_ARGUMENT). Release 빌드에서는 assert 가
  // 꺼져 segfault 로 나타나므로 반드시 만든다.
  iree_async_proactor_pool_t* pool = NULL;
  if (iree_status_is_ok(status)) {
    // ★ 베어메탈: runner.create 를 NULL 로 두면 pool 이 폴링 스레드를 만들지
    //   않는다 (proactor_pool.h: "When create is NULL ... proactors are created
    //   without a runner and the caller is responsible for polling").
    //   기본 옵션은 스레드 러너를 쓰므로 단일 코어에서 iree_thread_create 가
    //   불려 UNIMPLEMENTED 로 죽는다.
    // 기본값을 받아서 runner 만 지운다. 통째로 0 으로 밀면 proactor 내부
    // 옵션(버퍼/오퍼레이션 풀 용량 등)까지 0 이 되어 힙을 폭식한다.
    iree_async_proactor_pool_options_t pool_options =
        iree_async_proactor_pool_options_default();
    memset(&pool_options.runner, 0, sizeof(pool_options.runner));
    status = iree_async_proactor_pool_create(
        /*node_count=*/1, /*node_ids=*/NULL, pool_options, host_allocator,
        &pool);
  }
  if (iree_status_is_ok(status)) device_step = 3;

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = pool;
  if (iree_status_is_ok(status)) {
    status = iree_hal_sync_device_create(identifier, &params, &create_params,
                                         /*loader_count=*/1, &loader,
                                         device_allocator, host_allocator,
                                         out_device);
  }
  capture_status(status);
  if (iree_status_is_ok(status)) device_step = 4;
  iree_async_proactor_pool_release(pool);
  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(loader);
  return status;
}

static iree_status_t run(void) {
  iree_allocator_t host = iree_allocator_system();
  iree_runtime_instance_t* instance = NULL;
  iree_hal_device_t* device = NULL;
  iree_runtime_session_t* session = NULL;
  iree_vm_module_t* module = NULL;
  iree_runtime_call_t call;
  memset(&call, 0, sizeof(call));
  int call_ready = 0;

#if defined(MODEL_USE_BYTECODE)
  vmfb_len = (uint32_t)(_binary_model_vmfb_end - _binary_model_vmfb_start);
  vmfb_w0 = ((const uint32_t*)(const void*)_binary_model_vmfb_start)[0];
  vmfb_w2 = ((const uint32_t*)(const void*)_binary_model_vmfb_start)[2];
#endif
  iree_runtime_instance_options_t opts;
  iree_runtime_instance_options_initialize(&opts);
  iree_status_t status = iree_runtime_instance_create(&opts, host, &instance);
  abi_off_a1 = (uint32_t)offsetof(iree_vm_abi_ICrD_t, a1);
  abi_sizeof = (uint32_t)sizeof(iree_vm_abi_ICrD_t);
  ref_sizeof = (uint32_t)sizeof(iree_vm_ref_t);
  ref_align  = (uint32_t)_Alignof(iree_vm_ref_t);
  if (iree_status_is_ok(status)) inference_stage = 1;

  if (iree_status_is_ok(status)) status = create_device(host, &device);
  if (iree_status_is_ok(status)) inference_stage = 2;

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }
  if (iree_status_is_ok(status)) inference_stage = 3;

  {
    iree_vm_instance_t* vmi = iree_runtime_instance_vm_instance(instance);
    type_fence = (uint32_t)(uintptr_t)iree_vm_instance_lookup_type(
        vmi, iree_make_cstring_view("hal.fence"));
    type_bufview = (uint32_t)(uintptr_t)iree_vm_instance_lookup_type(
        vmi, iree_make_cstring_view("hal.buffer_view"));
    type_device = (uint32_t)(uintptr_t)iree_vm_instance_lookup_type(
        vmi, iree_make_cstring_view("hal.device"));
    type_buffer = (uint32_t)(uintptr_t)iree_vm_instance_lookup_type(
        vmi, iree_make_cstring_view("hal.buffer"));
  }

  // ★ EmitC: VMFB 를 읽는 대신 컴파일된 C 모듈을 만든다.
  if (iree_status_is_ok(status)) {
    status = model_module_create(iree_runtime_instance_vm_instance(instance), host,
                           &module);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_module(session, module);
  }
  if (iree_status_is_ok(status)) inference_stage = 4;

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(
        session, iree_make_cstring_view(MODEL_FUNCTION_NAME), &call);
    call_ready = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) inference_stage = 5;

  for (iree_host_size_t i = 0;
       i < MODEL_INPUT_COUNT && iree_status_is_ok(status); ++i) {
    const model_input_t* in = &model_inputs[i];
    iree_hal_buffer_view_t* view = NULL;
    status = iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), in->rank, in->shape,
        in->etype, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(in->data, in->bytes), &view);
    if (iree_status_is_ok(status)) {
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, view);
    }
    iree_hal_buffer_view_release(view);
  }
  if (iree_status_is_ok(status)) inference_stage = 6;

  if (iree_status_is_ok(status)) {
    uint64_t t0 = rdcycle();
    cyc_setup = (uint32_t)t0;                 // 여기까지가 준비 구간
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    cyc_invoke = (uint32_t)(rdcycle() - t0);  // 추론 구간만
    capture_status(status);
  }
  if (iree_status_is_ok(status)) inference_stage = 7;

  {
    iree_host_size_t off = 0;
    for (iree_host_size_t i = 0;
         i < MODEL_OUTPUT_COUNT && iree_status_is_ok(status); ++i) {
      iree_hal_buffer_view_t* out_view = NULL;
      status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &out_view);
      if (iree_status_is_ok(status)) {
        status = iree_hal_device_transfer_d2h(
            device, iree_hal_buffer_view_buffer(out_view), 0,
            &inference_output[off], model_output_elems[i] * sizeof(float),
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
      }
      if (iree_status_is_ok(status)) {
        float sum = 0.0f;
        for (iree_host_size_t k = 0; k < model_output_elems[i]; ++k) {
          sum += inference_output[off + k];
        }
        inference_out_sum[i] = sum;
        off += model_output_elems[i];
        inference_out_count = (uint32_t)(i + 1);
      }
      iree_hal_buffer_view_release(out_view);
    }
  }
  if (iree_status_is_ok(status)) inference_stage = 8;

  if (iree_status_is_ok(status)) {
    int32_t best = 0;
    for (int32_t i = 1; i < (int32_t)MODEL_OUTPUT0_ELEMS; ++i) {
      if (inference_output[i] > inference_output[best]) best = i;
    }
    inference_argmax = best;
    inference_max_value = inference_output[best];
    inference_stage = 9;
  }

  capture_status(status);
  if (call_ready) iree_runtime_call_deinitialize(&call);
  iree_vm_module_release(module);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  return status;
}


// --- 힙 실측 프로브 ---
extern char __heap_start__;
extern char __heap_end__;
extern void* _sbrk(int incr);
volatile uint32_t hp_start = 0, hp_end = 0, hp_cur = 0;
volatile uint32_t hp_sbrk_ok = 0, hp_malloc_ok = 0, hp_calloc_ok = 0;
volatile uint32_t hp_malloc_ptr = 0, hp_calloc_ptr = 0;

static void heap_probe(void) {
  hp_start = (uint32_t)(uintptr_t)&__heap_start__;
  hp_end   = (uint32_t)(uintptr_t)&__heap_end__;
  void* cur = _sbrk(0);
  hp_cur = (uint32_t)(uintptr_t)cur;
  void* g = _sbrk(64);
  hp_sbrk_ok = (g != (void*)-1) ? 1u : 0u;
  void* m = malloc(64);
  hp_malloc_ok = m ? 1u : 0u;
  hp_malloc_ptr = (uint32_t)(uintptr_t)m;
  void* c = calloc(1, 64);
  hp_calloc_ok = c ? 1u : 0u;
  hp_calloc_ptr = (uint32_t)(uintptr_t)c;
}

int main(void) {

  iree_status_t status = run();
  if (iree_status_is_ok(status)) {
    inference_status = 0;
    return 0;
  }
  inference_status_code = (uint32_t)iree_status_code(status);
  inference_status = 1;
  iree_status_free(status);
  return 1;
}
