let {PythonShell} = require('python-shell')

function create_gesture() {
  var folder = document.getElementById("path").value;
  var gesture = document.getElementById("gesture").value;
  console.log(folder);
  folder.replace(/\\/gm, " ")
  console.log(gesture);
  var options = {
    scriptPath : 'E:/an4/sem II/licenta/electron-quick-start/engine/',
    args : [folder, gesture]
  }

  let pyshell = new PythonShell('LanguageSignRec.py', options);


  pyshell.on('message', function(message) {
    console.log("message")
    swal('Frame no.'+ message);
  })

}



