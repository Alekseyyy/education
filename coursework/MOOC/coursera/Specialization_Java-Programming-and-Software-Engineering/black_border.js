/* A solution to part 3 of "Programming Exercise: Advanced Modifying Images" from Week 2
 * By Aleksey
 * 
 * The problem: "Write a function named setBlack that has one parameter pixel (representing a 
 * single pixel) and returns pixel with its red, green, and blue components changed so that 
 * the pixel’s color is black. Now you will write another function named addBorder. This function 
 * will add a black border to an image, [...] Work through the seven steps to write this function. 
 * Work an example by hand and note the steps you took before translating your algorithm to code. 
 * Which pixels should be part of the border? How will you identify those pixels? Once you have 
 * identified them, how will you make them black?"
 */

var sImage = new SimpleImage("smallpanda.png");

function addBorder(sImage, thickness) {
    for (var px of sImage.values()) {

        var x = px.getX();
        var y = px.getY();
        
        const setBlack = function() {
            px.setRed(0);
            px.setGreen(0);
            px.setBlue(0);
        }

        if (x >= 0 && x <= thickness)
            setBlack();
        if (y >= 0 && y <= thickness)
            setBlack();
        if (x >= sImage.getWidth() - thickness && x <= sImage.getWidth())
            setBlack();
        if (y >= sImage.getHeight() - thickness && y <= sImage.getHeight())
            setBlack();
    }
    return sImage;
}

print(addBorder(sImage, 10))
