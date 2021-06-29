/* A solution to part 2 of "Programming Exercise: Advanced Modifying Images" from Week 2
 * By Aleksey
 *
 * The Problem: "Your friend is trying to write a program that draws a square 200 pixels by 
 * 200 pixels and that looks like this square with colors red (red value 255), green (green 
 * value 255), blue (blue value 255) and magenta (red value 255 and blue value 255). All 
 * other RGB values are set to 0. [...] Your friend has written code to try to solve this problem, 
 * however their code has a bug [...] Your task is to find and fix the bug. Use what you have learned 
 * about applying the scientific method to debugging: gather information, apply your knowledge 
 * about images and programming, form a hypothesis, test your hypothesis, and finally, change the code 
 * to fix the problem."
 */

var img = new SimpleImage(200,200);

for (var px of img.values()){
  var x = px.getX();
  var y = px.getY();
  
  if (x <= img.getWidth()/2 && y <= img.getHeight()/2)
    px.setRed(255);
  if (x >= img.getWidth()/2 && y <= img.getHeight()/2)
      px.setGreen(255);
  if (x <= img.getWidth()/2 && y >= img.getHeight()/2){
    px.setRed(255);
    px.setBlue(255);
  }
  if (x >= img.getWidth()/2 && y >= img.getHeight()/2)
    px.setBlue(255);
}

print (img);
