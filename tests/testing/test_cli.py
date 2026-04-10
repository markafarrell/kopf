import pytest

from kopf.testing import KopfCLI

# KopfCLI spawns a real child process via multiprocessing with 'spawn' start method.
# Process startup is slow compared to threads, so we need generous timeouts.
pytestmark = pytest.mark.timeout(5)


def test_help_command():
    with KopfCLI(['--help']) as runner:
        pass
    assert runner.exit_code == 0
    assert 'Usage:' in runner.output


def test_run_subcommand_help():
    with KopfCLI(['run', '--help']) as runner:
        pass
    assert runner.exit_code == 0
    assert 'Usage:' in runner.output


def test_string_args_parsed():
    with KopfCLI('--help') as runner:
        pass
    assert runner.exit_code == 0
    assert 'Usage:' in runner.output


def test_bad_command_raises_with_reraise():
    with pytest.raises(RuntimeError, match="exited with code"):
        with KopfCLI(['nonexistent-command']):
            pass


def test_bad_command_suppressed_without_reraise():
    with KopfCLI(['nonexistent-command'], reraise=False) as runner:
        pass
    assert runner.exit_code == 2
    assert 'No such command' in runner.output


def test_reraise_chains_onto_block_exception():
    with pytest.raises(RuntimeError, match="exited with code") as exc_info:
        with KopfCLI(['nonexistent-command']):
            raise ValueError("block error")
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_block_exception_propagates_on_normal_exit():
    with pytest.raises(ValueError, match="block error"):
        with KopfCLI(['--help']):
            raise ValueError("block error")


def test_bad_syntax_file_fails(tmp_path):
    path = tmp_path / 'handlers.py'
    path.write_text("""This is a Python syntax error!""")
    with KopfCLI(['run', str(path), '--standalone'], reraise=False) as runner:
        pass
    assert runner.exit_code == 1
    assert 'SyntaxError' in runner.output


def test_timeout_forces_kill_and_raises(tmp_path):
    path = tmp_path / 'handlers.py'
    path.write_text("import time; time.sleep(5)")
    with pytest.raises(RuntimeError, match="did not exit gracefully"):
        with KopfCLI(['run', str(path), '--standalone'], exit_timeout=0.5) as runner:
            pass
    assert runner.exit_code == -9  # SIGKILL


def test_timeout_forces_kill_without_reraise(tmp_path):
    path = tmp_path / 'handlers.py'
    path.write_text("import time; time.sleep(5)")
    with KopfCLI(['run', str(path), '--standalone'], exit_timeout=0.5, reraise=False) as runner:
        pass
    assert runner.exit_code == -9  # SIGKILL
