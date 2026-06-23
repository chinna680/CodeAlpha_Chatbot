from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is artificial intelligence AI": "Artificial Intelligence (AI) is the simulation of human intelligence by machines to perform tasks like learning, reasoning, and problem-solving.",
    "What is machine learning ML": "Machine Learning is a subset of AI where machines learn from data automatically.",
    "What is Python programming language": "Python is a popular programming language used in AI, machine learning, and data science.",
    "What is YOLO object detection algorithm": "YOLO (You Only Look Once) is a real-time object detection algorithm used in computer vision.",
    "What is NLP natural language processing": "NLP helps computers understand and generate human language.",
    "What is a chatbot bot conversation AI": "A chatbot is an AI program that simulates conversation with humans for customer service.",
    "What is deep learning neural network": "Deep Learning uses neural networks with many layers to analyze images, text and audio.",
    "What is OpenCV image processing": "OpenCV is an open-source library for real-time computer vision and image processing.",
    "What is GitHub code repository": "GitHub is a platform for hosting and sharing code used by developers worldwide.",
    "What is CodeAlpha internship company": "CodeAlpha is a software company offering internship programs in AI and other tech domains.",
}

questions = list(faqs.keys())
answers = list(faqs.values())
vectorizer = TfidfVectorizer(ngram_range=(1,2))
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_input):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_index = similarities.argmax()
    if similarities[0][best_index] < 0.05:
        return "Sorry, I don't know that. Try asking differently!"
    return answers[best_index]

print("=" * 40)
print("   CodeAlpha FAQ Chatbot")
print("=" * 40)
print("Ask about AI, Python, YOLO, NLP etc.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break
    if user_input.strip() == "":
        continue
    print(f"Chatbot: {get_answer(user_input)}\n")