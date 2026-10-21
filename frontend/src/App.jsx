import React, { useEffect, useRef, useState } from 'react';

const EXAMPLES = ['纯牛奶整箱250ml', '男士轻便跑步运动鞋', '无线蓝牙降噪耳机', '保湿补水面霜50g'];

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { ...options, signal: options.signal || AbortSignal.timeout(60000) });
  } catch (error) {
    if (error.name === 'TimeoutError') throw new Error('请求超时，请稍后重试。');
    if (error.name === 'AbortError') throw error;
    throw new Error('无法连接分类服务，请确认后端已启动。');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof data?.detail === 'string' ? data.detail : '分类服务暂时不可用，请稍后重试。');
  }
  if (!data) throw new Error('服务返回了无效结果，请检查后端连接。');
  return data;
}

export default function App() {
  const [input, setInput] = useState('');
  const [health, setHealth] = useState(null);
  const [connection, setConnection] = useState('checking');
  const [labels, setLabels] = useState([]);
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [elapsed, setElapsed] = useState(null);
  const [showNeighbors, setShowNeighbors] = useState(false);
  const [retrieval, setRetrieval] = useState(null);
  const [retrievalChecking, setRetrievalChecking] = useState(false);
  const inputRef = useRef(null);
  const requestRef = useRef(null);
  const texts = input.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const validation = texts.length > 32 ? '每次最多分类 32 条标题，请减少输入。'
    : texts.some((text) => text.length > 2000) ? '单条标题不能超过 2000 个字符。' : '';

  async function checkService(signal) {
    setConnection('checking');
    try {
      const [status, categories] = await Promise.all([
        api('/health', { signal: signal || AbortSignal.timeout(10000) }),
        api('/labels', { signal: signal || AbortSignal.timeout(10000) }),
      ]);
      if (signal?.aborted) return;
      setHealth(status);
      setLabels(categories.labels);
      setConnection('online');
    } catch (error) {
      if (signal?.aborted) return;
      setConnection('offline');
      setHealth(null);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    checkService(controller.signal);
    return () => { controller.abort(); requestRef.current?.abort(); };
  }, []);

  useEffect(() => {
    if (!showNeighbors) return;
    const controller = new AbortController();
    setRetrievalChecking(true);
    setRetrieval(null);
    api('/retrieval/health', { signal: controller.signal })
      .then((status) => { if (!controller.signal.aborted) setRetrieval(status); })
      .catch((error) => {
        if (!controller.signal.aborted) setRetrieval({ status: 'unavailable', detail: error.message });
      })
      .finally(() => { clearTimeout(timeout); if (!controller.signal.aborted) setRetrievalChecking(false); });
    const timeout = setTimeout(() => {
      setRetrieval({ status: 'unavailable', detail: '加载检索超时，请稍后重新开启此选项。' });
      setRetrievalChecking(false);
      controller.abort();
    }, 60000);
    return () => { clearTimeout(timeout); controller.abort(); };
  }, [showNeighbors]);

  async function classify(event) {
    event.preventDefault();
    if (!texts.length || validation || busy) return;
    setError('');
    setResults([]);
    setElapsed(null);
    setBusy(true);
    const start = performance.now();
    const controller = new AbortController();
    requestRef.current = controller;
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const data = await api(showNeighbors ? '/analyze' : '/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts }), signal: controller.signal,
      });
      if (!Array.isArray(data.predictions) || data.predictions.length !== texts.length) {
        throw new Error('预测结果不完整，请重试。');
      }
      setResults(data.predictions);
      setElapsed(((performance.now() - start) / 1000).toFixed(2));
      if (connection !== 'online') checkService();
    } catch (error) {
      setError(error.name === 'AbortError' ? '请求已超时，请稍后重试。' : error.message);
    } finally {
      clearTimeout(timeout);
      setBusy(false);
      requestRef.current = null;
    }
  }

  function fillExamples() {
    setInput(EXAMPLES.join('\n'));
    setError('');
    inputRef.current?.focus();
  }

  function clear() {
    setInput('');
    setResults([]);
    setError('');
    setElapsed(null);
    inputRef.current?.focus();
  }

  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(results, null, 2)], { type: 'application/json;charset=utf-8' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'classification-results.json';
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>商品分类</h1>
        <a href="/docs" target="_blank" rel="noreferrer">API 文档</a>
      </header>
      <main>
        <p className="description">每行输入一个商品标题，点击“开始分类”查看结果。</p>
        <div className="workspace">
          <section aria-labelledby="input-heading">
            <div className="section-header">
              <h2 id="input-heading"><label htmlFor="product-titles">商品标题</label></h2>
              <button type="button" onClick={fillExamples} disabled={busy}>填入示例</button>
            </div>
            <form onSubmit={classify}>
              <textarea
                id="product-titles"
                ref={inputRef}
                value={input}
                onChange={(event) => { setInput(event.target.value); setError(''); }}
                disabled={busy}
                placeholder="请输入商品"
                aria-describedby="input-help"
                spellCheck="false"
              />
              <div className="input-bottom">
                <span id="input-help" className={validation ? 'invalid' : ''}>
                  {validation || '每次最多 32 条，单条最多 2000 字符'}
                </span>
                <span>{texts.length} / 32</span>
              </div>
              {error && <p className="error-message" role="alert">{error}</p>}
              <div className="retrieval-options">
                <label><input type="checkbox" checked={showNeighbors} disabled={busy} onChange={(event) => { setShowNeighbors(event.target.checked); setError(''); }} /> 显示相似商品</label>
                {showNeighbors && retrievalChecking && <span role="status">正在加载本地检索…</span>}
                {showNeighbors && !retrievalChecking && retrieval?.status === 'ready' && <span>索引已就绪，共 {retrieval.count} 条商品</span>}
                {showNeighbors && !retrievalChecking && retrieval?.status === 'unavailable' && <p className="error-message" role="alert">{retrieval.detail} 可关闭“显示相似商品”继续分类。</p>}
              </div>
              <div className="form-actions">
                <button type="submit" className="primary-button" disabled={busy || !texts.length || !!validation || (showNeighbors && (retrievalChecking || retrieval?.status !== 'ready'))}>
                  {busy ? '正在分类…' : '开始分类'}
                </button>
                <button type="button" onClick={clear} disabled={busy || (!input && !results.length)}>清空</button>
              </div>
            </form>
          </section>

          <section className="results-section" aria-labelledby="results-heading" aria-busy={busy}>
            <div className="section-header">
              <h2 id="results-heading">分类结果</h2>
              {results.length > 0 && <button type="button" onClick={download}>导出 JSON</button>}
            </div>
            <div aria-live="polite">
              {busy ? <p className="placeholder">正在分类，请稍候。</p>
                : results.length > 0 ? (
                  <>
                    <p className="result-summary">共 {results.length} 条，耗时 {elapsed} 秒</p>
                    <div className="table-wrapper">
                      <table>
                        <thead><tr><th scope="col">商品标题</th><th scope="col">类别</th><th scope="col">置信分数</th></tr></thead>
                        <tbody>{results.map((item, index) => (
                          <tr key={`${index}-${item.text}`}>
                            <td>
                              {item.text}
                              {Array.isArray(item.neighbors) && (
                                <details className="neighbors">
                                  <summary>查看相似商品（{item.neighbors.length}）</summary>
                                  {item.neighbors.length > 0 ? <ol>{item.neighbors.map((neighbor) => (
                                    <li key={neighbor.source_id}>
                                      <div>{neighbor.text}</div>
                                      <div className="neighbor-meta">类别：{neighbor.label} · 余弦相似度：{neighbor.similarity.toFixed(3)}</div>
                                      <small className="source-id" title={neighbor.source_id}>来源：{neighbor.source_id}</small>
                                    </li>
                                  ))}</ol> : <p>暂无参考商品。</p>}
                                  <p className="help-text">参考样本来自训练集，不改变 BERT 分类结果；相似度不是分类置信分数。</p>
                                </details>
                              )}
                            </td>
                            <td>{item.label}</td>
                            <td>{(item.confidence * 100).toFixed(1)}%</td>
                          </tr>
                        ))}</tbody>
                      </table>
                    </div>
                    <p className="help-text">置信分数为模型的 softmax 输出，不代表预测正确率。</p>
                  </>
                ) : <p className="placeholder">暂无结果。</p>}
            </div>
          </section>
        </div>

        {labels.length > 0 && (
          <details className="categories">
            <summary>支持的类别（{labels.length}）</summary>
            <p>{labels.join('、')}</p>
          </details>
        )}
      </main>
      <footer>
        <span role="status">
          {connection === 'online' ? `服务已连接 · ${String(health.device).toUpperCase()}`
            : connection === 'checking' ? '正在连接服务…' : '服务未连接，请确认后端已启动。'}
        </span>
        <button type="button" onClick={() => checkService()} disabled={connection === 'checking'}>重新连接</button>
      </footer>
    </div>
  );
}
