const path = require('path');

module.exports = env => ({
  entry: './App.js',
  resolve: {
    alias: {
      './bimsurfer': path.resolve(__dirname, '../bimsurfer/bimsurfer'),
      'three': path.resolve(__dirname, '../bimsurfer/bimsurfer/lib/three/r140/three.module.js'),
    }
  },
  output: {
    filename: `App.${env.postfix}.js`,
    path: path.resolve(__dirname, '.'),
  },
});
 