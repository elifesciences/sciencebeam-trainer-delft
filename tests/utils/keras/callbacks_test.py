from sciencebeam_trainer_delft.utils.keras.callbacks import ResumableEarlyStopping


class TestResumableEarlyStopping:
    def test_should_accept_no_initial_meta(self):
        callback = ResumableEarlyStopping()
        assert callback.initial_wait == 0
        assert callback.initial_stopped_epoch == 0
        assert callback.initial_best is None

    def test_should_accept_load_from_initial_meta(self):
        callback = ResumableEarlyStopping(
            initial_meta={
                ResumableEarlyStopping.MetaKeys.EARLY_STOPPING: {
                    ResumableEarlyStopping.MetaKeys.WAIT: 1,
                    ResumableEarlyStopping.MetaKeys.STOPPED_EPOCH: 2,
                    ResumableEarlyStopping.MetaKeys.BEST: 3
                }
            }
        )
        assert callback.initial_wait == 1
        assert callback.initial_stopped_epoch == 2
        assert callback.initial_best == 3

    def test_should_add_early_stopping_meta_to_logs(self):
        callback = ResumableEarlyStopping()
        callback.wait = 1
        callback.stopped_epoch = 2
        callback.best = 3
        logs = {}
        callback._add_early_stopping_meta_to_logs(logs)  # pylint: disable=protected-access
        assert logs[ResumableEarlyStopping.MetaKeys.EARLY_STOPPING] == {
            ResumableEarlyStopping.MetaKeys.WAIT: 1,
            ResumableEarlyStopping.MetaKeys.STOPPED_EPOCH: 2,
            ResumableEarlyStopping.MetaKeys.BEST: 3
        }
