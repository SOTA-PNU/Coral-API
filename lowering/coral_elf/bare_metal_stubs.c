// 베어메탈용 최소 구현 — IREE 체크아웃을 수정하지 않기 위한 것.
//
// 문제
// ----
// 이 IREE 리비전은 알 수 없는 플랫폼(IREE_PLATFORM_GENERIC)을 POSIX 로 간주한다.
// 두 곳 모두 else 로 떨어진다:
//
//   runtime/src/iree/async/CMakeLists.txt:105
//       else()  list(APPEND _platform_platform_deps iree::async::platform::posix)
//   runtime/src/iree/async/proactor_platform.c:54
//       #else   status = iree_async_proactor_create_posix(...)
//
// posix proactor 는 <sys/socket.h>, <netinet/in.h>, <unistd.h> 에 의존해
// Coral 에서 빌드 자체가 불가능하다.
//
// 왜 no-op proactor 로 충분한가
// -----------------------------
// local-sync 디바이스는 proactor 를 세마포어 생성에만 넘기고
// (sync_device.c:454), 세마포어는 그 포인터를 저장만 한다
// (async/semaphore.c:32). vtable 은 import_fence/export_fence 에서만dkR
// 호출되는데, 이는 외부 디바이스 fence 연동 경로라 우리 실행 경로에는 없다.
//
// 즉 "유효한 proactor 객체가 존재할 것"이 요구사항의 전부다. 실제 비동기
// I/O 는 하지 않는다. 호출되면 UNIMPLEMENTED 를 돌려주므로 가정이 틀리면
// 조용히 잘못 도는 대신 명확히 실패한다.
//
// IREE 가 GENERIC 플랫폼을 제대로 지원하게 되면 이 파일은 삭제할 수 있다.

#include "iree/base/api.h"

#if defined(IREE_PLATFORM_GENERIC)

#include <stdlib.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/async/platform/posix/api.h"
#include "iree/async/proactor.h"
#include "iree/base/threading/thread.h"

// 도달하면 안 되는 스텁 경로가 실제로 불렸는지 세는 카운터.
volatile uint32_t stub_hits = 0;
#define HIT(...) (stub_hits++)

//===--------------------------------------------------------------------===//
// no-op proactor
//===--------------------------------------------------------------------===//

typedef struct {
  iree_async_proactor_t base;
} iree_async_proactor_noop_t;

static void noop_destroy(iree_async_proactor_t* proactor) {
  iree_allocator_free(proactor->allocator, proactor);
}

static iree_async_proactor_capabilities_t noop_query_capabilities(
    iree_async_proactor_t* proactor) {
  HIT(9);
  (void)proactor;
  return 0;  // 아무 기능도 제공하지 않는다
}

volatile char stub_last_name[32];

static iree_status_t bm_unimpl(const char* name) {
  stub_hits++;
  for (int i = 0; i < 31; ++i) {
    stub_last_name[i] = name[i];
    if (!name[i]) break;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "bare-metal stub: unavailable on this platform");
}

#define NOOP_UNIMPL_UNUSED(name)                                                 \
  iree_make_status(IREE_STATUS_UNIMPLEMENTED,                             \
                   "proactor." name " is unavailable on bare-metal; "     \
                   "local-sync must not reach this path")

static iree_status_t noop_submit(iree_async_proactor_t* proactor,
                                 iree_async_operation_list_t operations) {
  HIT(0);
  (void)proactor, (void)operations;
  return bm_unimpl("submit");
}

static iree_status_t noop_poll(iree_async_proactor_t* proactor,
                               iree_timeout_t timeout,
                               iree_host_size_t* out_completed_count) {
  HIT(8);
  (void)proactor, (void)timeout;
  if (out_completed_count) *out_completed_count = 0;
  return iree_ok_status();  // 완료할 일이 없다 = 정상
}

static void noop_wake(iree_async_proactor_t* proactor) { (void)proactor; }

static iree_status_t noop_cancel(iree_async_proactor_t* proactor,
                                 iree_async_operation_t* operation) {
  HIT(1);
  (void)proactor, (void)operation;
  return bm_unimpl("cancel");
}

static iree_status_t noop_create_socket(iree_async_proactor_t* proactor,
                                        iree_async_socket_type_t type,
                                        iree_async_socket_options_t options,
                                        iree_async_socket_t** out_socket) {
  HIT(2);
  (void)proactor, (void)type, (void)options;
  if (out_socket) *out_socket = NULL;
  return bm_unimpl("create_socket");
}

static iree_status_t noop_import_socket(iree_async_proactor_t* proactor,
                                        iree_async_primitive_t primitive,
                                        iree_async_socket_type_t type,
                                        iree_async_socket_flags_t flags,
                                        iree_async_socket_t** out_socket) {
  HIT(3);
  (void)proactor, (void)primitive, (void)type, (void)flags;
  if (out_socket) *out_socket = NULL;
  return bm_unimpl("import_socket");
}

static void noop_destroy_socket(iree_async_proactor_t* proactor,
                                iree_async_socket_t* socket) {
  (void)proactor, (void)socket;
}

static iree_status_t noop_import_file(iree_async_proactor_t* proactor,
                                      iree_async_primitive_t primitive,
                                      iree_async_file_t** out_file) {
  HIT(4);
  (void)proactor, (void)primitive;
  if (out_file) *out_file = NULL;
  return bm_unimpl("import_file");
}

static void noop_destroy_file(iree_async_proactor_t* proactor,
                              iree_async_file_t* file) {
  (void)proactor, (void)file;
}

static iree_status_t noop_create_event(iree_async_proactor_t* proactor,
                                       iree_async_event_t** out_event) {
  HIT(5);
  (void)proactor;
  if (out_event) *out_event = NULL;
  return bm_unimpl("create_event");
}

static void noop_destroy_event(iree_async_proactor_t* proactor,
                               iree_async_event_t* event) {
  (void)proactor, (void)event;
}

static iree_status_t noop_register_event_source(
    iree_async_proactor_t* proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  HIT(6);
  (void)proactor, (void)handle, (void)callback;
  if (out_event_source) *out_event_source = NULL;
  return bm_unimpl("register_event_source");
}

static void noop_unregister_event_source(
    iree_async_proactor_t* proactor, iree_async_event_source_t* event_source) {
  (void)proactor, (void)event_source;
}

// local-sync 디바이스 생성이 이걸 실제로 부른다
// (sync_device.c:107 -> iree_async_notification_create).
//
// 단일 코어라 알림을 기다릴 다른 실행 주체가 없다. signal 은 epoch 를 올리고
// wait 은 그 epoch 를 보고 즉시 돌아오면 된다. 따라서 fd/futex 없이 epoch
// 원자변수만 있으면 충분하다. posix 백엔드의 create_notification
// (platform/posix/proactor.c:3297) 에서 fd 부분만 뺀 형태다.
static iree_status_t noop_create_notification(
    iree_async_proactor_t* proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  HIT(7);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;

  iree_async_notification_t* n = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(proactor->allocator, sizeof(*n), (void**)&n));
  memset(n, 0, sizeof(*n));

  iree_atomic_ref_count_init(&n->ref_count);
  n->proactor = proactor;
  iree_atomic_store(&n->epoch, 0, iree_memory_order_release);
  n->epoch_ptr = &n->epoch;
  n->flags = flags;
  // FUTEX 모드: epoch 원자변수만 쓰고 fd 를 요구하지 않는다.
  n->mode = IREE_ASYNC_NOTIFICATION_MODE_FUTEX;

  *out_notification = n;
  return iree_ok_status();
}


//===--------------------------------------------------------------------===//
// notification / 등록 계열 — vtable 의 나머지 절반
//===--------------------------------------------------------------------===//
//
// 이 멤버들을 비워두면 designated initializer 때문에 NULL 이 되고,
// NULL 함수 포인터 호출은 PC 를 0 으로 보낸다. Coral 링커 스크립트에서
// 0 번지는 _start 라, 프로그램이 조용히 처음부터 다시 시작하는 무한 루프가
// 된다 (칩에서 실측: 40 분 넘게 안 끝나고 inference_stage 가 1 로 되돌아감).
// 그래서 하나도 빠짐없이 채운다.

static void noop_destroy_notification(
    iree_async_proactor_t* proactor, iree_async_notification_t* notification) {
  if (notification) iree_allocator_free(proactor->allocator, notification);
}

static void noop_notification_signal(iree_async_proactor_t* proactor,
                                     iree_async_notification_t* notification,
                                     int32_t wake_count) {
  (void)proactor, (void)wake_count;
  if (!notification) return;
  // 깨울 상대가 없으므로 epoch 만 올린다. 대기 측은 이 값을 본다.
  iree_atomic_fetch_add(&notification->epoch, 1, iree_memory_order_release);
}

// 단일 코어라 대기 중에 값을 바꿔줄 다른 실행 주체가 없다. local_sync 는
// 완전히 동기적이어서 signal 이 wait 보다 먼저 일어나므로, 여기서는
// "이미 신호됨"으로 답해야 한다. false 를 돌려주면 호출자가 영원히 돈다.
static bool noop_notification_wait(iree_async_proactor_t* proactor,
                                   iree_async_notification_t* notification,
                                   uint32_t wait_token,
                                   iree_timeout_t timeout) {
  (void)proactor, (void)notification, (void)wait_token, (void)timeout;
  return true;
}

static iree_status_t noop_create_notification_shared(
    iree_async_proactor_t* proactor,
    const iree_async_notification_shared_options_t* options,
    iree_async_notification_t** out_notification) {
  (void)proactor, (void)options;
  HIT();
  if (out_notification) *out_notification = NULL;
  return bm_unimpl("create_notification_shared");
}

static iree_status_t noop_register_relay(
    iree_async_proactor_t* proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  (void)proactor, (void)source, (void)sink, (void)flags, (void)error_callback;
  HIT();
  if (out_relay) *out_relay = NULL;
  return bm_unimpl("register_relay");
}

static void noop_unregister_relay(iree_async_proactor_t* proactor,
                                  iree_async_relay_t* relay) {
  (void)proactor, (void)relay;
}

static iree_status_t noop_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)proactor, (void)state, (void)buffer, (void)access_flags;
  HIT();
  if (out_entry) *out_entry = NULL;
  return bm_unimpl("register_buffer");
}

static iree_status_t noop_register_dmabuf(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)proactor, (void)state, (void)dmabuf_fd, (void)offset, (void)length,
      (void)access_flags;
  HIT();
  if (out_entry) *out_entry = NULL;
  return bm_unimpl("register_dmabuf");
}

static void noop_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  (void)proactor, (void)entry, (void)state;
}

static iree_status_t noop_register_slab(
    iree_async_proactor_t* proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  (void)proactor, (void)slab, (void)access_flags;
  HIT();
  if (out_region) *out_region = NULL;
  return bm_unimpl("register_slab");
}

static iree_status_t noop_import_fence(iree_async_proactor_t* proactor,
                                       iree_async_primitive_t fence,
                                       iree_async_semaphore_t* semaphore,
                                       uint64_t signal_value) {
  (void)proactor, (void)fence, (void)semaphore, (void)signal_value;
  HIT();
  return bm_unimpl("import_fence");
}

static iree_status_t noop_export_fence(iree_async_proactor_t* proactor,
                                       iree_async_semaphore_t* semaphore,
                                       uint64_t wait_value,
                                       iree_async_primitive_t* out_fence) {
  (void)proactor, (void)semaphore, (void)wait_value;
  HIT();
  return bm_unimpl("export_fence");
}

static void noop_set_message_callback(
    iree_async_proactor_t* proactor,
    iree_async_proactor_message_callback_t callback) {
  (void)proactor, (void)callback;
}

static iree_status_t noop_send_message(iree_async_proactor_t* target,
                                       uint64_t message_data) {
  (void)target, (void)message_data;
  HIT();
  return bm_unimpl("send_message");
}

static iree_status_t noop_subscribe_signal(
    iree_async_proactor_t* proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback,
    iree_async_signal_subscription_t** out_subscription) {
  (void)proactor, (void)signal, (void)callback;
  HIT();
  if (out_subscription) *out_subscription = NULL;
  return bm_unimpl("subscribe_signal");
}

static void noop_unsubscribe_signal(
    iree_async_proactor_t* proactor,
    iree_async_signal_subscription_t* subscription) {
  (void)proactor, (void)subscription;
}

static const iree_async_proactor_vtable_t iree_async_proactor_noop_vtable = {
    .destroy = noop_destroy,
    .query_capabilities = noop_query_capabilities,
    .submit = noop_submit,
    .poll = noop_poll,
    .wake = noop_wake,
    .cancel = noop_cancel,
    .create_socket = noop_create_socket,
    .import_socket = noop_import_socket,
    .destroy_socket = noop_destroy_socket,
    .import_file = noop_import_file,
    .destroy_file = noop_destroy_file,
    .create_event = noop_create_event,
    .destroy_event = noop_destroy_event,
    .register_event_source = noop_register_event_source,
    .unregister_event_source = noop_unregister_event_source,
    .create_notification = noop_create_notification,
    .create_notification_shared = noop_create_notification_shared,
    .destroy_notification = noop_destroy_notification,
    .notification_signal = noop_notification_signal,
    .notification_wait = noop_notification_wait,
    .register_relay = noop_register_relay,
    .unregister_relay = noop_unregister_relay,
    .register_buffer = noop_register_buffer,
    .register_dmabuf = noop_register_dmabuf,
    .unregister_buffer = noop_unregister_buffer,
    .register_slab = noop_register_slab,
    .import_fence = noop_import_fence,
    .export_fence = noop_export_fence,
    .set_message_callback = noop_set_message_callback,
    .send_message = noop_send_message,
    .subscribe_signal = noop_subscribe_signal,
    .unsubscribe_signal = noop_unsubscribe_signal,
};

// proactor_platform.c 의 #else 분기가 이 이름을 부른다.
iree_status_t iree_async_proactor_create_posix(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  (void)options;
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  iree_async_proactor_noop_t* proactor = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, sizeof(*proactor),
                                             (void**)&proactor));
  iree_async_proactor_initialize(&iree_async_proactor_noop_vtable,
                                 IREE_SV("bare-metal-noop"), allocator,
                                 &proactor->base);
  *out_proactor = &proactor->base;
  return iree_ok_status();
}

//===--------------------------------------------------------------------===//
// threading
//===--------------------------------------------------------------------===//
// IREE_ENABLE_THREADING=OFF 라 구현이 빌드되지 않지만 async/proactor_thread.c
// 가 참조한다. 단일 코어이므로 스레드 생성은 정의상 실패한다.

iree_status_t iree_thread_create(iree_thread_entry_t entry, void* entry_arg,
                                 iree_thread_create_params_t params,
                                 iree_allocator_t allocator,
                                 iree_thread_t** out_thread) {
  HIT(10);
  (void)entry, (void)entry_arg, (void)params, (void)allocator;
  if (out_thread) *out_thread = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "threads are unavailable on bare-metal");
}

void iree_thread_release(iree_thread_t* thread) { (void)thread; }

//===--------------------------------------------------------------------===//
// 힙 할당기
//===--------------------------------------------------------------------===//
//
// 왜 직접 만드는가
// ----------------
// 이 툴체인의 newlib 할당기는 이 보드에서 제대로 동작하지 않는다. 칩에서 실측:
//
//   * aligned_alloc(64, 1024) == NULL, memalign(64, 1024) == NULL
//     -> 정렬 할당 계열이 아예 없다. IREE 의 iree_aligned_alloc
//        (base/internal/memory.c:291) 이 여기에 걸린다.
//   * _sbrk 는 4 KB 만 받아간 뒤 다시 부르지 않는데 calloc(242) 가 NULL 을
//     돌려준다 (같은 시점에 calloc(276) 은 성공). 즉 힙을 확장해야 할 때
//     확장하지 않는다.
//
// 그 결과가 iree/base/allocator_libc.c:57 의
//   "RESOURCE_EXHAUSTED; libc allocator failed the request"
// 이고, iree_hal_sync_device_create 가 여기서 죽었다.
//
// 어떻게
// ------
// __heap_start__ ~ __heap_end__ (링커 스크립트상 DDR 64 MB) 위에 first-fit +
// 인접 블록 병합 방식의 단순 할당기를 올린다. malloc/calloc/realloc/free 는
// --wrap 으로, aligned_alloc 은 강한 정의로 대체한다. IREE 는 물론 newlib
// 내부(printf 등)의 호출까지 전부 이쪽으로 들어온다.
//
// 모든 반환 포인터 바로 앞 4바이트에 블록 헤더까지의 오프셋을 적어두므로
// free() 는 정렬 할당이든 아니든 동일하게 원본 블록을 찾을 수 있다.

#include <stdint.h>
#include <string.h>

extern char __heap_start__, __heap_end__;

#define BM_HDR 32u          // 헤더 크기 (16바이트 정렬 유지)
#define BM_MIN_ALIGN 16u

typedef struct bm_blk {
  uint32_t size;            // 페이로드 바이트 수
  uint32_t is_free;
  struct bm_blk* next;      // 물리적 다음 블록
  struct bm_blk* prev;      // 물리적 이전 블록
  uint32_t reserved[2];
} bm_blk_t;

static char* bm_lo = 0;     // .bss -> CRT 가 0 으로 미는 것이 보장된다
static char* bm_hi = 0;
static char* bm_brk = 0;
static bm_blk_t* bm_head = 0;

// 진단용 (시뮬레이터에서 덤프)
volatile uint32_t bm_allocs = 0;
volatile uint32_t bm_frees = 0;
volatile uint32_t bm_fail = 0;
volatile uint32_t bm_fail_size = 0;
volatile uint32_t bm_peak_kb = 0;

static void bm_init(void) {
  if (bm_lo) return;
  bm_lo = &__heap_start__;
  bm_hi = &__heap_end__;
  bm_brk = bm_lo;
  bm_head = 0;
}

static uint32_t bm_align_up(uint32_t v, uint32_t a) {
  return (v + a - 1u) & ~(a - 1u);
}

// alignment 를 만족하는 사용자 포인터를 가진 블록을 돌려준다.
static void* bm_alloc_aligned(uint32_t size, uint32_t alignment) {
  bm_init();
  if (size == 0) size = 1;
  if (alignment < BM_MIN_ALIGN) alignment = BM_MIN_ALIGN;

  // 정렬 패딩 + 오프셋 워드까지 감안한 최대 필요량
  uint32_t need = bm_align_up(size, BM_MIN_ALIGN) + alignment + BM_MIN_ALIGN;

  bm_blk_t* blk = 0;
  for (bm_blk_t* b = bm_head; b; b = b->next) {
    if (b->is_free && b->size >= need) { blk = b; break; }
  }

  if (blk) {
    // 남는 부분이 충분히 크면 쪼갠다
    uint32_t rest = blk->size - need;
    if (rest > BM_HDR + BM_MIN_ALIGN * 2u) {
      bm_blk_t* nb = (bm_blk_t*)((char*)blk + BM_HDR + need);
      nb->size = rest - BM_HDR;
      nb->is_free = 1;
      nb->next = blk->next;
      nb->prev = blk;
      if (blk->next) blk->next->prev = nb;
      blk->next = nb;
      blk->size = need;
    }
    blk->is_free = 0;
  } else {
    // 힙 확장
    if (bm_brk + BM_HDR + need > bm_hi) {
      bm_fail++;
      if (!bm_fail_size) bm_fail_size = size;
      return 0;
    }
    blk = (bm_blk_t*)bm_brk;
    blk->size = need;
    blk->is_free = 0;
    blk->next = 0;
    blk->prev = 0;
    if (bm_head) {
      bm_blk_t* last = bm_head;
      while (last->next) last = last->next;
      last->next = blk;
      blk->prev = last;
    } else {
      bm_head = blk;
    }
    bm_brk += BM_HDR + need;
    uint32_t used_kb = (uint32_t)(bm_brk - bm_lo) / 1024u;
    if (used_kb > bm_peak_kb) bm_peak_kb = used_kb;
  }

  // 사용자 포인터: 헤더 뒤에서 정렬하되 오프셋 워드 자리를 반드시 남긴다
  uintptr_t base = (uintptr_t)blk + BM_HDR + sizeof(uint32_t);
  uintptr_t user = (base + (alignment - 1u)) & ~(uintptr_t)(alignment - 1u);
  ((uint32_t*)user)[-1] = (uint32_t)(user - (uintptr_t)blk);

  bm_allocs++;
  return (void*)user;
}

static void bm_free(void* ptr) {
  if (!ptr) return;
  bm_init();
  if ((char*)ptr < bm_lo || (char*)ptr >= bm_hi) return;  // 우리 힙이 아니다

  uint32_t off = ((uint32_t*)ptr)[-1];
  bm_blk_t* blk = (bm_blk_t*)((char*)ptr - off);
  if ((char*)blk < bm_lo || (char*)blk >= bm_brk) return;
  if (blk->is_free) return;                                // 이중 해제
  blk->is_free = 1;
  bm_frees++;

  // 인접 블록 병합
  bm_blk_t* n = blk->next;
  if (n && n->is_free) {
    blk->size += BM_HDR + n->size;
    blk->next = n->next;
    if (n->next) n->next->prev = blk;
  }
  bm_blk_t* p = blk->prev;
  if (p && p->is_free) {
    p->size += BM_HDR + blk->size;
    p->next = blk->next;
    if (blk->next) blk->next->prev = p;
  }
}

// 사용자 포인터에서 실제로 쓸 수 있는 바이트 수
static uint32_t bm_usable(void* ptr) {
  uint32_t off = ((uint32_t*)ptr)[-1];
  bm_blk_t* blk = (bm_blk_t*)((char*)ptr - off);
  return blk->size - (off - BM_HDR);
}

void* __wrap_malloc(size_t size) {
  return bm_alloc_aligned((uint32_t)size, BM_MIN_ALIGN);
}

void* __wrap_calloc(size_t n, size_t size) {
  uint32_t total = (uint32_t)(n * size);
  void* p = bm_alloc_aligned(total, BM_MIN_ALIGN);
  if (p) memset(p, 0, total);
  return p;
}

void* __wrap_realloc(void* ptr, size_t size) {
  if (!ptr) return bm_alloc_aligned((uint32_t)size, BM_MIN_ALIGN);
  if (size == 0) { bm_free(ptr); return 0; }
  uint32_t old = bm_usable(ptr);
  if (old >= (uint32_t)size) return ptr;
  void* np = bm_alloc_aligned((uint32_t)size, BM_MIN_ALIGN);
  if (!np) return 0;
  memcpy(np, ptr, old);
  bm_free(ptr);
  return np;
}

void __wrap_free(void* ptr) { bm_free(ptr); }

// newlib 것을 강한 정의로 덮는다 (IREE 의 iree_aligned_alloc 이 부른다)
void* aligned_alloc(size_t alignment, size_t size) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) return 0;
  return bm_alloc_aligned((uint32_t)size, (uint32_t)alignment);
}

void* memalign(size_t alignment, size_t size) {
  return aligned_alloc(alignment, size);
}

int posix_memalign(void** out_ptr, size_t alignment, size_t size) {
  void* p = aligned_alloc(alignment, size);
  if (!p) return 12;  // ENOMEM
  *out_ptr = p;
  return 0;
}

#endif  // IREE_PLATFORM_GENERIC
