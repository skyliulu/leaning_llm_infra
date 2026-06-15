window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],
    displayMath: [['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: '.*',
    processHtmlClass: 'arithmatex',
  },
  startup: {
    pageReady: () =>
      window.MathJax.startup.defaultPageReady().then(() => {
        scheduleMathJaxRender();
      }),
  },
};

let mathJaxRenderTimer = 0;

function scheduleMathJaxRender() {
  clearTimeout(mathJaxRenderTimer);
  mathJaxRenderTimer = window.setTimeout(renderMathJax, 50);
}

function renderMathJax() {
  if (!window.MathJax || typeof window.MathJax.typesetPromise !== 'function') {
    return;
  }

  if (typeof window.MathJax.typesetClear === 'function') {
    window.MathJax.typesetClear([document.body]);
  }

  window.MathJax.typesetPromise([document.body]).catch((error) => {
    console.error('MathJax rendering failed:', error);
  });
}

if (typeof document$ !== 'undefined') {
  document$.subscribe(scheduleMathJaxRender);
} else {
  document.addEventListener('DOMContentLoaded', scheduleMathJaxRender);
}

window.addEventListener('hashchange', scheduleMathJaxRender);
