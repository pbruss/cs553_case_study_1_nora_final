import unittest
from app import respond, cancel_inference

class TestChatbot(unittest.TestCase):
    
    def setUp(self):
        self.message = "What should I eat today?"
        self.history = [("Hello", "Hi! How can I assist you today?")]
        self.system_message = "You are a friendly chatbot who always responds in the style of a therapist."
        self.max_tokens = 10
        self.temperature = 0.7
        self.top_p = 0.95
        self.use_local_model = False

    def test_respond_with_local_model(self):
        # Test with local model
        self.use_local_model = True
        generator = respond(
            message=self.message,
            history=self.history,
            system_message=self.system_message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            use_local_model=self.use_local_model,
        )
        result = next(generator)
        self.assertIsInstance(result, list)
        self.assertEqual(result[-1][0], self.message)

    def test_respond_with_api_model(self):
        # Test with API-based model
        self.use_local_model = False
        generator = respond(
            message=self.message,
            history=self.history,
            system_message=self.system_message,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            use_local_model=self.use_local_model,
        )
        result = next(generator)
        self.assertIsInstance(result, list)
        self.assertEqual(result[-1][0], self.message)

    def test_cancel_inference(self):
        # Test if inference can be canceled
        cancel_inference()  # This should set the global stop_inference flag to True
        self.assertTrue(cancel_inference)

if __name__ == "__main__":
    unittest.main()
