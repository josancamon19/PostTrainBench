"""Monkey-patch Harbor's ModalEnvironment to resolve [environment.env] from task.toml.

Harbor's Modal environment ignores task_env_config.env (the [environment.env] section
in task.toml). The Daytona environment resolves these correctly, but Modal doesn't.
This patch adds the same resolution logic to Modal's start() method.

Usage:
    python -c "import src.harbor_patch" && harbor run ...
    # or
    python src/harbor_patch.py  # patches and then forwards to harbor CLI
"""


def apply():
    from harbor.environments.modal import ModalEnvironment

    _original_start = ModalEnvironment.start

    async def _patched_start(self, force_build):
        # Resolve [environment.env] and store for injection into sandbox secrets
        if self.task_env_config.env:
            from harbor.utils.env import resolve_env_vars

            self._resolved_task_env = resolve_env_vars(self.task_env_config.env)
        else:
            self._resolved_task_env = {}

        return await _original_start(self, force_build)

    _original_create_sandbox = ModalEnvironment._create_sandbox.__wrapped__

    async def _patched_create_sandbox(self, gpu_config, secrets_config, volumes_config):
        from modal import Secret

        if getattr(self, "_resolved_task_env", None):
            secrets_config = list(secrets_config) + [Secret.from_dict(self._resolved_task_env)]
        return await _original_create_sandbox(self, gpu_config, secrets_config, volumes_config)

    # Apply patches
    ModalEnvironment.start = _patched_start

    # _create_sandbox is wrapped with tenacity @retry, need to patch the inner function
    import tenacity

    _patched_create_sandbox_retried = tenacity.retry(
        stop=tenacity.stop_after_attempt(2),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )(_patched_create_sandbox)
    ModalEnvironment._create_sandbox = _patched_create_sandbox_retried


apply()

if __name__ == "__main__":
    from harbor.cli.main import app

    app()
