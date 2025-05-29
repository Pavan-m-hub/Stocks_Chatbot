// Initialize the chatbot with a welcome message
document.addEventListener('DOMContentLoaded', () => {
  console.log("DOM fully loaded");
  const chatBox = document.getElementById('chat-box');
  
  if (!chatBox) {
    console.error("Chat box element not found!");
    alert("Error: Chat box element not found!");
    return;
  }
  
  console.log("Chat box found, adding welcome message");
  
  // Add welcome message
  const welcomeMessage = `
    Welcome to the Stock Market Predictor! 👋
    <br><br>
    I can help you with:
    <br>
    • Stock price information
    <br>
    • Stock movement predictions
    <br>
    • Market sentiment analysis
    <br>
    • Stock news recommendations
    <br><br>
    Try asking me about a specific stock like AAPL, MSFT, GOOGL, or about the market in general!
  `;
  
  addBotMessage(welcomeMessage);
  
  // Focus on input field
  const inputField = document.getElementById('user-input');
  if (inputField) {
    inputField.focus();
  } else {
    console.error("Input field not found!");
    alert("Error: Input field not found!");
  }
});

// Send message to the chatbot
async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  
  if (!message) return;
  
  // Clear input field
  input.value = '';
  
  // Add user message to chat
  addUserMessage(message);
  
  // Show loading indicator
  const loadingId = showLoading();
  
  try {
    console.log("Sending request to backend:", message);
    // Send request to backend
    const response = await fetch('http://localhost:8080/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    
    // Hide loading indicator
    hideLoading(loadingId);
    
    // Process response
    const data = await response.json();
    
    if (data.error) {
      // Show error message
      addErrorMessage(data.error);
    } else {
      // Add bot response to chat
      addBotMessage(data.response);
    }
  } catch (error) {
    // Hide loading indicator
    hideLoading(loadingId);
    
    // Show error message
    addErrorMessage('Unable to connect to the server. Please try again later.');
    console.error('Error:', error);
  }
  
  // Scroll to bottom of chat
  scrollToBottom();
}

// Use a suggestion
function usesuggestion(text) {
  const input = document.getElementById('user-input');
  input.value = text;
  sendMessage();
}

// Add user message to chat
function addUserMessage(message) {
  const chatBox = document.getElementById('chat-box');
  const messageElement = document.createElement('div');
  messageElement.className = 'message user-message';
  messageElement.textContent = message;
  chatBox.appendChild(messageElement);
  scrollToBottom();
}

// Add bot message to chat
function addBotMessage(message) {
  const chatBox = document.getElementById('chat-box');
  const messageElement = document.createElement('div');
  messageElement.className = 'message bot-message';
  messageElement.innerHTML = message;
  chatBox.appendChild(messageElement);
  scrollToBottom();
}

// Add error message to chat
function addErrorMessage(message) {
  const chatBox = document.getElementById('chat-box');
  const errorElement = document.createElement('div');
  errorElement.className = 'error-message';
  errorElement.textContent = message;
  chatBox.appendChild(errorElement);
  scrollToBottom();
}

// Show loading indicator
function showLoading() {
  const chatBox = document.getElementById('chat-box');
  const loadingElement = document.createElement('div');
  loadingElement.className = 'message bot-message loading';
  loadingElement.id = 'loading-' + Date.now();
  loadingElement.innerHTML = `
    Analyzing
    <div class="loading-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `;
  chatBox.appendChild(loadingElement);
  scrollToBottom();
  return loadingElement.id;
}

// Hide loading indicator
function hideLoading(loadingId) {
  const loadingElement = document.getElementById(loadingId);
  if (loadingElement) {
    loadingElement.remove();
  }
}

// Scroll to bottom of chat
function scrollToBottom() {
  const chatBox = document.getElementById('chat-box');
  chatBox.scrollTop = chatBox.scrollHeight;
}