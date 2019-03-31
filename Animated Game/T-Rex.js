function TRex(x, y, width,height) {

	this.x = x;
	this.y = y;

	this.yVelocity = 0;
	this.speed = 1;
	this.onGround = true;
    this.width=width;
    this.height=height;
}

/**
	*	handle y values
	*/
TRex.prototype.update = function(platform) {

	var bottom = this.y + this.height; // bottom pixel of circle
	var nextBottom = bottom + this.yVelocity; // calculate next frame's bottom

  if (bottom <= platform && nextBottom >= platform) { // next frame will be on platform

		this.yVelocity = 0; // reset velocity
		this.y = platform - this.height; // don't go past platform
		this.onGround = true;
  } else if (platform - bottom > 1) { // nowhere near platform

		this.yVelocity += this.speed; // increase velocity
		this.onGround = false;
  }

	/* movement */
	this.y += this.yVelocity;
};

/**
	* make the dino jump
	*/
TRex.prototype.jump = function() {

	this.yVelocity = -(this.height * 0.275); // jump
};

TRex.prototype.draw = function() {

  //fill(125);
   // stroke(125);
  //strokeWeight(0);
  //noStroke();
  image(img,this.x, this.y,this.width,this.height);
  //circle(this.x, this.y, this.radius*2);
};