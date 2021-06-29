/* A solution to part 3 of "Programming Exercise: Modifying Images" from Week 2
 * By Aleksey
 * 
 * The problem: "Write code to change the Duke blue devil [...] to be yellow."
 */

var sImage = new SimpleImage("duke_blue_devil.png");

for (var px of sImage.values()) {
    var x = px.getX();
    var y = px.getY();
    if (!(px.getRed() === 255 && px.getGreen() === 255 && px.getBlue() === 255)) {
        px.setRed(255);
        px.setGreen(255);
        px.setBlue(0);
    }
}

print(sImage);
