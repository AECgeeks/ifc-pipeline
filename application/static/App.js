require(["bimsurfer/src/MultiModal", "bimsurfer/lib/domReady!"], function (Viewer) {
    var v = new Viewer({
        domNode: 'right',
        svgDomNode: 'bottom',
        modelId: window.MODEL_ID,
        withTreeVisibilityToggle: true,
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
    
    if (window.CHECK_ID) {    
        var loader = new THREE.GLTFLoader();
        loader.load(`/run/${window.CHECK_ID}/result/resource/gltf/0.glb`, function(gltf) {
            v.bimSurfer3D.viewer.scene.add(gltf.scene);
        });
    }
});
