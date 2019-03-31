var horizon;
var obstacleSpeed;
var score;
var obstacles = [];
var dino;
var img;

function setup() {
  createCanvas(1200, 400);
  textAlign(CENTER);
  img = loadImage('image/dino.jpg');
    horizon = height - 40;
    score = 0;
	obstacleSpeed = 10;

	var size = 20;
    var xPos = 50;
    var yPos = 0;
    var dinoWidth = 60;
    var dinoHeight = 60;
    //dino = new TRex(size * 2, height - horizon, size);
    dino = new TRex(xPos, yPos, dinoWidth, dinoHeight);
  textSize(20);
}

function draw() {
    background(224);
    drawHUD();
	handleLevel(frameCount);
	dino.update(horizon);
    handleObstacles();
}

/**
	* draws horizon & score
	*/
function drawHUD() {

  /* draw horizon */
  stroke(128);
	strokeWeight(1);
  line(0, horizon, width, horizon);

	/* draw score */
    noStroke();
    fill(1);
  text("Score: " + score, width / 2, 30);

	/* draw T-Rex */
    dino.draw();
}

/**
	*	updates, draws, and cleans out the obstacles
	*/
function handleObstacles() {

  for (var i = obstacles.length - 1; i >= 0; i--) {

		obstacles[i].update(obstacleSpeed);
		obstacles[i].draw();

		if (obstacles[i].hits(dino)) // if there's a collision
			endGame();

    if (!obstacles[i].onScreen) // if it's no longer showing
      obstacles.splice(i, 1); // delete from array
  }
}


/**
	* speeds game up, pushes new obstacles, & handles score
	*/
function handleLevel(n) {

  if (n % 30 === 0) { // every 0.5 seconds

    var n = noise(n); // noisey

    if (n > 0.5)
      newObstacle(n); // push new obstacle

	  if (n % 60 === 0) // every 2 seconds
	    obstacleSpeed *= 1555; // speed up
  }

	score++;
}

/**
	* pushes random obstacle
	*/
function newObstacle(n) {

	var col = 1;
    var size_y = 50+random(20) ;
    var size_x=15+random(10);
  var obs = new Obstacle(width, size_x,size_y, horizon, col);
 
  obstacles.push(obs);
}

function keyPressed() {

	if ((keyCode === UP_ARROW || keyCode === 32) && dino.onGround) // jump if possible
		dino.jump();
}



function endGame() {

	noLoop();
  noStroke();
  textSize(40);
  text("GAME OVER", width / 2, height / 2);
  //textSize(20);
  //text("Press f5 to restart", width / 2, height / 2 + 20);
}