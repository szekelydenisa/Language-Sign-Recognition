function speak_mode() {
  console.log("start")
  var options = {
    scriptPath : 'E:/an4/sem II/licenta/electron-quick-start/engine/',
    args: '-m Speak'
  }

  PythonShell.run('finalPrediction.py', options, function (err, results) {
    if (err) console.log(err);
    // results is an array consisting of messages collected during execution
    console.log('results: %j', results);
  });

}



