
/* A solution to part 1 of "Programming Exercise: Modifying Images" from Week 2
 * By Aleksey
 * 
 * The problem: "Write a JavaScript program that modifies an image by putting three 
 * vertical stripes on it - a red stripe on the left one third, a green stripe in the 
 * middle, and a blue stripe on the right one third. For example, if your program ran 
 * on Drew’s picture shown on the left, the resulting image would have red, green and 
 * blue vertical stripes as shown in the image on the right."
 */

var sImage = new SimpleImage("hilton.jpg");

for (var px of sImage.values()) {
    var x = px.getX();

    if (x >= 0 && x <= sImage.getWidth() * (1/3))
        px.setRed(255);
    if (x > sImage.getWidth() * (1/3) && x <= sImage.getWidth() * (2/3))
        px.setGreen(255);
    if (x > sImage.getWidth() * (2/3) && x <= sImage.getWidth())
        px.setBlue(255);
}

print (sImage);
