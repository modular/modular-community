from std.testing import assert_true
from filelock_mojo import FileLock, ReadWriteLock, LockErrorKind


def main() raises:
    var owner = FileLock("installed-package.lock", timeout=0)
    var rival = FileLock("installed-package.lock", timeout=0)
    with owner.hold():
        assert_true(owner.is_locked())
        assert_true(not rival.try_acquire())
    assert_true(rival.try_acquire())
    rival.release()
    var reader = FileLock("read.lock", timeout=0, mode=.shared)
    with reader.hold():
        pass
    var rw = ReadWriteLock("readwrite.lock", timeout=0)
    with rw.read():
        pass
    with rw.write():
        pass
    var rejected = False
    try:
        rw.release()
    except e:
        rejected = e.kind == LockErrorKind.not_held
    assert_true(rejected)
    var attempted = owner.try_hold()
    assert_true(Bool(attempted))
    with attempted.take():
        assert_true(owner.lock_count() == 1)
    assert_true(not owner.is_locked())
    print("PASS: installed filelock_mojo")
