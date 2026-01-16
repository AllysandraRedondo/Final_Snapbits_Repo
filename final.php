<!DOCTYPE html>
<html>
<head>
    <title>Finishing it up!</title>
    <style>

body {
    height: 100vh; 
    margin: 0;
}

.header {
    display: flex;
    padding: 20px;
    padding-bottom: 25px;
    height: 40px;
    margin: 0;
    background-color: #3d2f6d;
    justify-content: space-between;
    align-items: center;
    text-align: center;
    
}

.Logo {
    height: 50px;
    width: 60px;
    margin-right: 20px;
    margin-left: 20px;
    display: flex;
}

.logo {
    height: 50px;
    width: auto;
    margin-right: 20px;
    margin-left: 20px;

}

ul {
    font-family: century gothic;
    color: white;
    margin-right: 30px;
    
}

li {
    list-style: none;
    display: inline-block;
    font-weight: bold;
}

li a {
    color: white;
    text-decoration: none;
}

.headText {
    font-family: century gothic;
    color: white;
    font-weight: bold;
}

.showing {
    padding: 10px;
    margin: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
}

h2 {
    text-align: center;
    font-family: century gothic;
    margin-top: 40px;
    margin-bottom: 20px;
}

    </style>
</head>
<body>
<div class="top">
    <div class="header">
        <div class="Logo">
            <img src="../Images/Logo.png" img class="logo" alt="Logo">
            <p class="headText">SnapBits!</p>
        </div>
    </div>
</div>
<div class="h1">
    <h2>Looking Good! Here is your Photo!</h2>
    <h2 id="countdown">
    <br><br>
</div>

<div id ="viewing" class = "showing">
    
<script>

  const img = document.createElement('img');

  const token = new URLSearchParams(window.location.search).get('token'); 
  img.src = `displayImage.php?token=${token}`;
  img.width = 300;
  img.alt = "Your image";

  document.getElementById('viewing').appendChild(img);

const viewing = document.getElementById('viewing');
const countDownTimer = document.getElementById('countdown');
let timeleft = 10;

const timeout = setInterval(() => {
    let minutes = Math.floor(timeleft /60);
    let seconds = timeleft % 60;

    countDownTimer.textContent = `You can only view this picture for ${minutes}:${seconds.toString().padStart(2, '0')} minutes`;

    if (timeleft <=0) {
    clearInterval(timeout);
    countDownTimer.textContent = "Your access is expired, thank you for trying SnapBits.";
    viewing.innerHTML = '';
    window.location.href = "redirection.html";
    }

    timeleft--;

}, 1000);

</script>


</div>

</body>
</html>