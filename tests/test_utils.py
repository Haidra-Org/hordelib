from hordelib.settings import UserSettings


class TestWorkerSettings:
    def test_worker_settings_singleton(self):
        a = UserSettings()
        b = UserSettings()
        assert a is b
