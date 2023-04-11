from hordelib.settings import WorkerSettings


class TestWorkerSettings:
    def test_worker_settings_singleton(self):
        a = WorkerSettings()
        b = WorkerSettings()
        assert a is b
