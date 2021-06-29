/* A solution to part 1 of "Programming Exercise: Advanced Modifying Images" from Week 2
 * By Aleksey, after Robert Duvall et al.
 * 
 * The problem: "Write the green screen algorithm you saw in the lecture video yourself. To 
 * make sure you really understand the code that was written in the video, you should write 
 * the code yourself without looking at the video unless you get stuck and need to refer back 
 * to it for a hint."
 */

var fImage = new SimpleImage("drewRobert.png");
var bImage = new SimpleImage("dinos.png");
var output = new SimpleImage(fImage.getWidth(), fImage.getHeight())

for (var pixel of fImage.values()) {
    if (pixel.getGreen() > pixel.getRed() + pixel.getBlue()) {
        var x = pixel.getX();
        var y = pixel.getY();
        var bgPixel = bImage.getPixel(x, y);
        output.setPixel(x, y, bgPixel);
    }
    else {
        output.setPixel(pixel.getX(), pixel.getY(), pixel);
    }
}

print (output);
