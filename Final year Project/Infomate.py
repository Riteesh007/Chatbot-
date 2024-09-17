import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class InfomateChatbot:
    def __init__(self, text_file):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.prompt = "You: "
        self.examples_embeddings = []  # Store precomputed embeddings
        self.example_responses = []  # Store example responses

        with open(text_file, 'r') as file:
            user_input = None
            response = None
            for line in file:
                line = line.strip()
                if line.startswith("Human: "):
                    if user_input is not None and response is not None:
                        self.examples_embeddings.append(self.model.encode([user_input])[0])
                        self.example_responses.append(response)
                    user_input = line.split(": ", 1)[1]
                elif line.startswith("Infomate: "):
                    response = line.split(": ", 1)[1]
            # Add the last example
            if user_input is not None and response is not None:
                self.examples_embeddings.append(self.model.encode([user_input])[0])
                self.example_responses.append(response)

    def generate_response(self, user_input):
        user_input_lower = user_input.lower()
        best_response = "I'm sorry, I don't understand that."

        # Encode user input
        user_embedding = self.model.encode([user_input_lower])[0]

        for example_embedding, example_response in zip(self.examples_embeddings, self.example_responses):
            # Calculate cosine similarity
            similarity = cosine_similarity([user_embedding], [example_embedding])[0][0]

            if similarity > 0.8:  # Adjust the threshold as needed
                best_response = example_response
                return best_response  # Return the matched response immediately

        return best_response  # Return the default response if no match is found

# Initialize the chatbot
text_file = "C:/Users/user/Downloads/Final year Project/database.txt"  # Update with the path to your text file
chatbot = InfomateChatbot(text_file)

# Interaction loop
print("Infomate Chatbot")
print("Type 'exit' to end the conversation.")

while True:
    user_input = input(chatbot.prompt)
    
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    response = chatbot.generate_response(user_input)
    print("Bot:", response)
