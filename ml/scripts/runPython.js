const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const mlRoot = path.resolve(__dirname, '..');
const backendRoot = path.resolve(mlRoot, '..');
const preferredVenvName = '.venv311';

const resolvePythonPath = () => {
  if (process.env.ML_PYTHON_PATH) {
    const configuredPath = process.env.ML_PYTHON_PATH;
    if (fs.existsSync(configuredPath)) {
      return configuredPath;
    }
  }

  const windowsVenv = path.join(mlRoot, preferredVenvName, 'Scripts', 'python.exe');
  if (fs.existsSync(windowsVenv)) {
    return windowsVenv;
  }

  const unixVenv = path.join(mlRoot, preferredVenvName, 'bin', 'python');
  if (fs.existsSync(unixVenv)) {
    return unixVenv;
  }

  return 'python';
};

const ensureVenv = () => {
  const python = resolvePythonPath();
  if (python !== 'python' && fs.existsSync(python)) {
    return python;
  }

  const windowsVenv = path.join(mlRoot, preferredVenvName, 'Scripts', 'python.exe');
  const create = process.platform === 'win32'
    ? spawnSync('py', ['-3.11', '-m', 'venv', preferredVenvName], { cwd: mlRoot, stdio: 'inherit' })
    : spawnSync('python3.11', ['-m', 'venv', preferredVenvName], { cwd: mlRoot, stdio: 'inherit' });
  if (create.status !== 0) {
    return 'python';
  }
  return fs.existsSync(windowsVenv) ? windowsVenv : resolvePythonPath();
};

const ensureRequirements = (python) => {
  const requirementsPath = path.join(mlRoot, 'requirements.txt');
  if (!fs.existsSync(requirementsPath)) {
    return;
  }

  const install = spawnSync(python, ['-m', 'pip', 'install', '--trusted-host', 'pypi.org', '--trusted-host', 'files.pythonhosted.org', '-r', requirementsPath], {
    cwd: mlRoot,
    stdio: 'inherit',
  });
  if (install.status !== 0) {
    process.exit(install.status ?? 1);
  }
};

const main = () => {
  const setupOnly = process.argv.includes('--setup');
  const scriptArgs = process.argv.slice(2).filter((arg) => arg !== '--setup');
  const python = ensureVenv();
  ensureRequirements(python);

  if (setupOnly) {
    console.log(`[ml] Python ready: ${python}`);
    return;
  }

  if (scriptArgs.length === 0) {
    console.error('Usage: node ml/scripts/runPython.js [--setup] <script.py> [args...]');
    process.exit(1);
  }

  const [script, ...args] = scriptArgs;
  const scriptPath = path.isAbsolute(script) ? script : path.join(mlRoot, script);
  const result = spawnSync(python, [scriptPath, ...args], {
    cwd: mlRoot,
    stdio: 'inherit',
    env: process.env,
  });
  process.exit(result.status ?? 1);
};

main();
