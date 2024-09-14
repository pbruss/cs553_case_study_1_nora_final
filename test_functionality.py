import unittest
from app import respond
from unittest.mock import patch, MagicMock

class TestChatbotResponse(unittest.TestCase):

    @patch('app.pipe')  # Mock the local inference pipeline
    @patch('app.client.chat_completion')  # Mock the API-based inference
    def test_respond_with_local_model(self, mock_client_chat_completion, mock_pipe):
        # Mock local model response
        mock_pipe.return_value = [{'generated_text': [{'content': 'response from local model'}]}]
        
        # History and message for testing
        history = [("User's previous question", "Previous bot response")]
        message = "New message from user"
        system_message = "You are a friendly chatbot."
        use_local_model = True

        # Simulate calling the generator function
        response_generator = respond(message, history, system_message=system_message, use_local_model=use_local_model)
        
        # Convert generator to a list
        responses = list(response_generator)

        # Validate the response content
        expected_response = history + [(message, "response from local model")]
        self.assertIn(expected_response, responses)

    @patch('app.pipe')  # Mock the local inference pipeline
    @patch('app.client.chat_completion')  # Mock the API-based inference
    def test_respond_with_api_model(self, mock_client_chat_completion, mock_pipe):
        # Mock API model response
        mock_client_chat_completion.return_value = [{'choices': [{'delta': {'content': 'response from API model'}}]}]
        
        # History and message for testing
        history = [("User's previous question", "Previous bot response")]
        message = "New message from user"
        system_message = "You are a friendly chatbot."
        use_local_model = False

        # Simulate calling the generator function
        response_generator = respond(message, history, system_message=system_message, use_local_model=use_local_model)
        
        # Convert generator to a list
        responses = list(response_generator)

        # Validate the response content
        expected_response = history + [(message, "response from API model")]
        self.assertIn(expected_response, responses)

if __name__ == '__main__':
    unittest.main()
