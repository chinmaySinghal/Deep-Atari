function Obstacle(x, size_x,size_y, horizon, color) {

  this.x = x;
	this.y = horizon - size_y;

  this.size_x = size_x;
  this.size_y= size_y;
  this.color = color;

	this.onScreen = true;
}

/**
	*	handle x and onScreen values
	*/
Obstacle.prototype.update = function(speed) {

	/* check if offscreen */
	this.onScreen = (this.x > -this.size_x);

	/* movement */
	this.x -= speed;
};

Obstacle.prototype.draw = function() {

	fill(this.color);
	//stroke(this.color);
    //strokeWeight(0);
    noStroke();
	rect(this.x, this.y, this.size_x, this.size_y,5,5,0,0);
};

/**
	* checks for collisions
	*/
Obstacle.prototype.hits = function(dino) {

    //var halfSize_x = this.size_x / 2;
    //var halfSize_y = this.size_y / 2;
    
    //var minimumDistance = halfSize_x + (dino.width); // closest before collision

    var minimumDistance = (this.size_x/2) + (dino.width/2); // closest before collision
	/* find center coordinates */
	var xCenter = this.x + this.size_x/2;
	var yCenter = this.y + this.size_y/2;

    //var distance = dist(xCenter, yCenter, dino.x, dino.y); // calculate distance from centers
    var distance = dist(xCenter, yCenter, dino.x+dino.width/2, dino.y+dino.height/2); // calculate distance from centers

	return (distance < minimumDistance); // return result
};
