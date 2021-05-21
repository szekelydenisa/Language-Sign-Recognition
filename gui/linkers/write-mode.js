let {PythonShell} = require('python-shell')
var path = require("path")

function write_mode() {
  loading = true;
  console.log("start")
  var options = {
    scriptPath : 'E:/an4/sem II/licenta/electron-quick-start/engine/',
  }
  PythonShell.run('finalPrediction.py', options, function (err, results) {
    if (err) console.log(err);
    // results is an array consisting of messages collected during execution
    console.log('results: %j', results);
  });

  
}



