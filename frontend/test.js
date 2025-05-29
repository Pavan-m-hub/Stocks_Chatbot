console.log("Test script loaded successfully!");

document.addEventListener('DOMContentLoaded', () => {
  const body = document.querySelector('body');
  const testDiv = document.createElement('div');
  testDiv.textContent = "If you can see this, JavaScript is working correctly.";
  testDiv.style.padding = "20px";
  testDiv.style.backgroundColor = "#f0f0f0";
  testDiv.style.margin = "20px";
  testDiv.style.borderRadius = "5px";
  body.appendChild(testDiv);
});