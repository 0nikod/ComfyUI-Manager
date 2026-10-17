"""PyTorch recovery with optional torchaudio; no package installs are executed."""
import tempfile
import unittest
from unittest.mock import patch

from manager_test_utils import load_manager_util


manager_util = load_manager_util()


class PyTorchRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.check_output = self.enterContext(
            patch.object(manager_util.subprocess, 'check_output', return_value=0)
        )
        self.make_pip_cmd = self.enterContext(
            patch.object(manager_util, 'make_pip_cmd', side_effect=lambda args: ['pip'] + args)
        )
        self.error = self.enterContext(patch.object(manager_util.logging, 'error'))

    def inventory(self, audio=False):
        packages = {'torch': '2.7.0+cu128', 'torchvision': '0.22.0+cu128',
                    'comfyui_frontend_package': '1.0.0'}
        if audio:
            packages['torchaudio'] = '2.7.0+cu128'
        return packages

    def recover(self, previous, current):
        with patch.object(manager_util, 'get_installed_packages', return_value=current) as installed:
            manager_util.PIPFixer(
                previous, self.directory.name, self.directory.name
            ).fix_broken()
        installed.assert_called_once_with(True)

    def test_unchanged_with_and_without_audio(self):
        for audio in (False, True):
            with self.subTest(audio=audio):
                previous = self.inventory(audio)
                self.recover(previous, previous.copy())
                self.error.assert_not_called()
                self.check_output.assert_not_called()

    def test_missing_required_packages(self):
        for package in ('torch', 'torchvision'):
            with self.subTest(package=package):
                previous = self.inventory(True)
                del previous[package]
                self.recover(previous, previous.copy())
                self.error.assert_called_once_with('[ComfyUI-Manager] PyTorch is not installed')
                self.check_output.assert_not_called()
                self.error.reset_mock()

    def test_required_changes_and_removals_trigger_recovery(self):
        for audio in (False, True):
            for package in ('torch', 'torchvision'):
                for version in ('0.0.0', None):
                    with self.subTest(audio=audio, package=package, version=version):
                        previous = self.inventory(audio)
                        current = previous.copy()
                        if version is None:
                            del current[package]
                        else:
                            current[package] = version
                        self.recover(previous, current)
                        expected = ['pip', 'install', 'torch==2.7.0', 'torchvision==0.22.0']
                        if audio:
                            expected.append('torchaudio==2.7.0')
                        expected += ['--index-url', 'https://download.pytorch.org/whl/cu128']
                        self.check_output.assert_called_once_with(expected, universal_newlines=True)
                        self.error.assert_not_called()
                        self.check_output.reset_mock()

    def test_audio_add_remove_and_change_do_not_trigger_recovery(self):
        for before, after in ((None, '2.7.0'), ('2.7.0', None), ('2.7.0', '2.8.0')):
            with self.subTest(before=before, after=after):
                previous, current = self.inventory(), self.inventory()
                if before:
                    previous['torchaudio'] = before
                if after:
                    current['torchaudio'] = after
                self.recover(previous, current)
                self.error.assert_not_called()
                self.check_output.assert_not_called()

    def test_rollback_platform_commands(self):
        for platform in ('cu128', 'rocm6.3', 'cpu', 'xpu'):
            for known in (True, False):
                for audio in (False, True):
                    with self.subTest(platform=platform, known=known, audio=audio):
                        version = '2.7.0' if known else '2.9.0.dev20250101'
                        manager_util.torch_rollback(f'{version}+{platform}', include_torchaudio=audio)
                        if known:
                            args = ['install', 'torch==2.7.0', 'torchvision==0.22.0']
                            if audio:
                                args.append('torchaudio==2.7.0')
                            index = f'https://download.pytorch.org/whl/{platform}'
                        else:
                            args = ['install', '--pre', 'torch', 'torchvision']
                            if audio:
                                args.append('torchaudio')
                            index = f'https://download.pytorch.org/whl/nightly/{platform}'
                        self.check_output.assert_called_once_with(
                            ['pip'] + args + ['--index-url', index], universal_newlines=True
                        )
                        self.error.assert_not_called()
                        self.check_output.reset_mock()

    def test_rollback_without_platform_and_legacy_default(self):
        for audio in (False, True):
            with self.subTest(audio=audio):
                if audio:
                    manager_util.torch_rollback('2.7.0')
                else:
                    manager_util.torch_rollback('2.7.0', include_torchaudio=False)
                cmd = ['pip', 'install', '--force', 'torch', 'torchvision']
                if audio:
                    cmd.append('torchaudio')
                self.check_output.assert_called_once_with(cmd, universal_newlines=True)
                self.error.assert_called_once_with(cmd)
                self.check_output.reset_mock()
                self.error.reset_mock()

    def test_snapshots_keep_protecting_optional_audio(self):
        for audio in (False, True):
            with self.subTest(audio=audio):
                snapshot = {'torch==2.7.0': '', 'torchvision==0.22.0': '',
                            'nvidia-cublas-cu12==12.8': '', 'example==1.0': ''}
                if audio:
                    snapshot['torchaudio==2.7.0'] = ''
                manager_util.restore_pip_snapshot(snapshot, ['--pip-non-url'])
                self.check_output.assert_called_once_with(['pip', 'install', 'example==1.0'])
                self.error.assert_not_called()
                self.check_output.reset_mock()


if __name__ == '__main__':
    unittest.main()
