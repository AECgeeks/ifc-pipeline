require(["bimsurfer/src/MultiModal", "bimsurfer/lib/domReady!"], function (Viewer) {
    var v = window.viewer = new Viewer({
        domNode: 'right',
        svgDomNode: 'bottom',
        modelId: window.MODEL_ID,
        withTreeVisibilityToggle: true,
        withTreeViewIcons: true,
        n_files: window.NUM_FILES
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
