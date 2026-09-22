"""UNIT — livesub.logging: LineLog mirroring, WorkLog JSONL schema, thread safety."""
import json
import threading

import pytest

from livesub.logging import LineLog, WorkLog


class TestLineLog:
    def test_writes_file_and_mirrors_stdout(self, tmp_path, capsys):
        log = LineLog(tmp_path / "live.txt")
        log.write("hello 字幕")
        log.close()
        out = capsys.readouterr().out
        assert "hello 字幕" in out
        text = (tmp_path / "live.txt").read_text(encoding="utf-8")
        assert "hello 字幕" in text
        # 每行带 HH:MM:SS.mmm 时间戳前缀
        assert text.split(" ")[0].count(":") == 2

    def test_parent_dir_is_callers_job(self, tmp_path):
        # 契约：LineLog 不建父目录（cli.py 里由调用方先 mkdir）；
        # WorkLog 自己建。两边行为不同是现状，勿在测试里"统一"。
        with pytest.raises(FileNotFoundError):
            LineLog(tmp_path / "nope" / "x.txt")
        (tmp_path / "nope").mkdir()
        log = LineLog(tmp_path / "nope" / "x.txt")
        log.write("ok")
        log.close()
        assert (tmp_path / "nope" / "x.txt").exists()

    def test_none_path_prints_only(self, capsys):
        log = LineLog(None)
        log.write("no file")
        assert "no file" in capsys.readouterr().out

    def test_close_is_idempotent(self, tmp_path):
        log = LineLog(tmp_path / "x.txt")
        log.write("a")
        log.close()
        log.close()  # must not raise


class TestWorkLog:
    def test_event_schema(self, tmp_path):
        work = WorkLog(tmp_path / "w.jsonl")
        work.event("asr", seg_id=1, ja="こんにちは")
        work.close()
        rec = json.loads((tmp_path / "w.jsonl").read_text(encoding="utf-8").strip())
        assert rec["kind"] == "asr"
        assert rec["seg_id"] == 1
        assert rec["ja"] == "こんにちは"
        assert "ts" in rec

    def test_japanese_not_escaped(self, tmp_path):
        work = WorkLog(tmp_path / "w.jsonl")
        work.event("mt", zh="蜻蜓点水")
        work.close()
        raw = (tmp_path / "w.jsonl").read_text(encoding="utf-8")
        assert "蜻蜓点水" in raw  # ensure_ascii=False

    def test_appends_multiple_events(self, tmp_path):
        work = WorkLog(tmp_path / "w.jsonl")
        for i in range(5):
            work.event("t", i=i)
        work.close()
        lines = (tmp_path / "w.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5

    def test_creates_parent_dirs(self, tmp_path):
        work = WorkLog(tmp_path / "deep" / "nest" / "w.jsonl")
        work.event("t")
        work.close()
        assert (tmp_path / "deep" / "nest" / "w.jsonl").exists()

    def test_concurrent_writers_do_not_interleave(self, tmp_path):
        work = WorkLog(tmp_path / "w.jsonl")

        def worker(n):
            for i in range(50):
                work.event("t", worker=n, i=i)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        work.close()
        lines = (tmp_path / "w.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 400
        for line in lines:  # every line must be intact JSON
            json.loads(line)
