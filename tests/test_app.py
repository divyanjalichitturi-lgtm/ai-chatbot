import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from chatbot import ChatbotError


class ChatUiTests(unittest.TestCase):
    def test_successful_submission_shows_assistant_response(self) -> None:
        with patch("chatbot.ask_bot", return_value="Courses include science."):
            app = AppTest.from_file("app.py").run()
            app.chat_input[0].set_value("What courses are offered?").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertIn(
            "Courses include science.",
            [item.value for item in app.markdown],
        )

    def test_failed_submission_can_be_retried_without_duplicate_message(self) -> None:
        with patch(
            "chatbot.ask_bot",
            side_effect=[ChatbotError("Model unavailable."), "Recovered answer."],
        ) as ask:
            app = AppTest.from_file("app.py").run()
            app.chat_input[0].set_value("What are the fees?").run()

            self.assertEqual([item.value for item in app.error], ["Model unavailable."])
            self.assertEqual(
                [item.label for item in app.button],
                ["Retry the last question"],
            )

            app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(ask.call_count, 2)
        markdown = [item.value for item in app.markdown]
        self.assertEqual(markdown.count("What are the fees?"), 1)
        self.assertIn("Recovered answer.", markdown)


if __name__ == "__main__":
    unittest.main()
