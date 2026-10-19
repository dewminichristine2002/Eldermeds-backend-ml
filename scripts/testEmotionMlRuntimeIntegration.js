const net = require('net');
const http = require('http');
const axios = require('axios');

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function main() {
  const occupiedPort = await freePort();
  const wrongService = http.createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ service: 'unrelated_service', ready: true }));
  });
  await new Promise((resolve) => wrongService.listen(occupiedPort, '127.0.0.1', resolve));
  process.env.EMOTION_ML_SERVICE_URL = `http://127.0.0.1:${occupiedPort}`;
  const wrongPortManager = require('../src/services/emotionMlServiceManager');
  let rejectedWrongPort = false;
  try { await wrongPortManager.ensureEmotionMlServiceAvailable(); }
  catch (error) { rejectedWrongPort = /not the ElderMeds emotion classifier/.test(error.message); }
  assert(rejectedWrongPort, 'A wrong service on the configured port should fail identity validation.');
  assert(wrongService.listening, 'The manager must leave an unrelated port occupant running.');
  await new Promise((resolve, reject) => wrongService.close((error) => error ? reject(error) : resolve()));
  console.log('wrong-port identity rejection: passed (unrelated process left running)');

  const port = await freePort();
  process.env.EMOTION_ML_SERVICE_URL = `http://127.0.0.1:${port}`;
  process.env.EMOTION_ML_STARTUP_TIMEOUT_MS = process.env.EMOTION_ML_STARTUP_TIMEOUT_MS || '120000';
  process.env.EMOTION_ML_STARTUP_POLL_MS = '500';
  delete require.cache[require.resolve('../src/services/emotionMlServiceManager')];
  const manager = require('../src/services/emotionMlServiceManager');
  const { requestEmotionPrediction } = require('../src/services/emotionPredictionService');

  try {
    const concurrent = await Promise.all([
      manager.ensureEmotionMlServiceAvailable(),
      manager.ensureEmotionMlServiceAvailable(),
    ]);
    assert(concurrent.every(Boolean), 'Concurrent ensures should share one successful startup.');

    const health = await axios.get(`${process.env.EMOTION_ML_SERVICE_URL}/health`, { timeout: 3000 });
    assert(health.data.service === 'emotion_classifier', 'Unexpected service identity.');
    assert(health.data.ready === true, 'Model readiness did not become true.');
    assert(health.data.model_version === 'minilm_logistic_regression_v4', 'Production v4 was not active.');
    assert(health.data.supported_classes.length === 7, 'Unexpected emotion class set.');

    const prediction = await requestEmotionPrediction('I felt happier after my daughter called.', { timeout: 60000 });
    assert(prediction.detectionSource === 'ml_model', 'Node did not accept the real service prediction.');
    assert(prediction.modelVersion === health.data.model_version, 'Node and API model versions differ.');
    process.env.EMOTION_ML_SERVICE_AUTO_START = 'false';
    await manager.stopEmotionMlService();
    const { analyzeNarrative } = require('../src/services/narrativeAnalysisService');
    const fallback = await analyzeNarrative('I felt lonely today.');
    assert(fallback.detectionSource === 'rule_fallback', 'Unavailable service must preserve Node rule fallback.');
    assert(fallback.fallbackReason === 'ml_unavailable', 'Fallback should preserve the unavailability reason.');
    console.log(JSON.stringify({
      startup: 'passed', concurrentEnsures: 'passed', readiness: 'passed', nodePrediction: 'passed',
      unavailableFallback: 'passed',
      service: health.data.service, model: health.data.model_version, port,
    }, null, 2));
  } finally {
    process.env.EMOTION_ML_SERVICE_AUTO_START = 'true';
    await manager.stopEmotionMlService();
    try {
      await axios.get(`${process.env.EMOTION_ML_SERVICE_URL}/health`, { timeout: 1000 });
      throw new Error('Owned emotion service remained reachable after shutdown.');
    } catch (error) {
      if (error.message === 'Owned emotion service remained reachable after shutdown.') throw error;
      console.log('shutdown: passed (owned service stopped)');
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
