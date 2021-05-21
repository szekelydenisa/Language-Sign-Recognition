function train_network() {
  var folder = document.getElementById("path").value;
  console.log(folder);
  folder.replace(/\\/gm, " ")
  console.log(folder);
  var options = {
    scriptPath : 'E:/an4/sem II/licenta/electron-quick-start/engine/',
    args : [folder]
  }

  let pyshell = new PythonShell('trainVGG.py', options);

  pyshell.on('message', function(message) {
    console.log("message")
    swal(message);
  });

}



