import MultiModal from './bimsurfer/src/MultiModal.js';

let processed = false;

document.addEventListener('DOMContentLoaded', () => {
  if (processed) {
    // For some reason the event fires twice with es-module-shims on chrome
    return;
  }
  processed = true;

  var v = window.viewer = new MultiModal({
    domNode: 'right',
    svgDomNode: 'bottom',
    modelId: window.MODEL_ID,
    withTreeVisibilityToggle: true,
    withTreeViewIcons: true,
    n_files: window.NUM_FILES,
    withShadows: true,
    engine3d: localStorage.getItem('engine') || 'threejs',
    fromPipeline: true
  });

  if (window.SPINNER_CLASS) {
    v.setSpinner({className: window.SPINNER_CLASS});
  } else if (window.SPINNER_URL) {
    v.setSpinner({url: window.SPINNER_URL});
  }
  
  v.load2d();
  v.load3d();
  v.loadMetadata('middle');
  v.loadTreeView('top');

  if (window.onViewerLoaded) {
    window.onViewerLoaded(self);
  }
});
