# NLP-A7 AIT GPT

This assignment is completed under the guidance of Professor Dr. Chaklam Silpasuwanchai in the AT82.05 Artificial Intelligence: Natural Language Understanding (NLU).

This is done by st124087 (Kyi Thin Nu)

## Section
- [Overview of this assignment](#overview-of-this-assignment)
- [ Task1: Source Discovery ](#task-1-source-discovery)
- [ Task2: Analysis and Problem Solving](#task-2-analysis-and-problem-solving)
- [ Task3: Chatbot Web App Development ](#task-3-chatbot-development---web-application-development)
- [Acknowledgements](#acknowledgments)

## Overview of this assignment

### Brief Introduction
In this assignment, I will create a chatbot, AIT-GPT, that specializes in answering questions related to the Asian Institute of Technology (AIT).

## Task 1:  Source Discovery
1) information sources related to AIT:
    For the sample information of AIT for this task, I take one of the brochure.pdf related to AIT.
2) Design a Prompt Template for AIT-GPT to handle questions related to AIT and develop a model that
can provide gentle and informative answers based on the designed template.

Proper credit is given to AIT for providing access to relevant information and resources related to the institution.

## Task 2: Analysis and Problem Solving
The model's performance in retrieving information was hindered by limitations in the training data, resulting in suboptimal performance. While the model showed potential in retrieving relevant documents and providing coherent responses, it struggled due to insufficient or inadequate training data.

To address these challenges and improve the model's performance, the following steps are recommended:

- Data augmentation: Augmenting the training data by incorporating diverse and representative examples can help the model learn a wider range of patterns and improve its generalization ability.

- Domain-specific training: Fine-tuning the model on domain-specific data relevant to the application can enhance its understanding of context and improve the relevance of retrieved information.

- Transfer learning: Leveraging pre-trained models and fine-tuning them on the target task or domain can accelerate the learning process and improve performance, especially when data is limited.

- Quality assurance: Implementing rigorous quality assurance measures to ensure the accuracy and relevance of the training data can prevent biases and errors from affecting the model's performance.

- Continuous evaluation and iteration: Continuously evaluating the model's performance against predefined metrics and iteratively refining its architecture and training approach based on feedback can lead to incremental improvements over time.

By addressing these limitations and adopting strategies to enhance the model's training data, it is possible to overcome its current shortcomings and achieve more satisfactory results in information retrieval tasks.

## Task 3. Chatbot Development - Web Application Development

The integration of the web application with a language model enables users to interact with a chatbot interface and receive coherent responses and relevant source documents in real-time. By following the outlined workflow and ensuring seamless integration, the web application provides an intuitive and efficient user experience for information retrieval tasks.

#### Technologies Used:

- Flask: A lightweight web framework for Python used to develop the backend of the application.
- HTML/CSS: Used for designing the user interface and styling the chat interface.
- JavaScript: Employed for enhancing user interaction and dynamic content rendering.
- Hugging Face Transformers: Leveraged for integrating advanced NLP models into the chatbot.
- Git: Version control system for collaboration and code management.

#### Implementation:

- Backend Development: The backend of the web application was developed using Flask, a Python web framework. Flask routes were defined to handle user requests and interactions with the chatbot model.
- Chatbot Model Integration: A pre-trained NLP model from Hugging Face Transformers was integrated into the application to power the chatbot. The model was capable of generating coherent responses to user queries.
- User Interface Design: The user interface was designed using HTML/CSS to create a chat interface where users could input messages and view responses. The interface was designed to be intuitive and user-friendly.
- Source Document Retrieval: Mechanisms were implemented to retrieve relevant source documents based on the user's queries. These documents were presented along with the chatbot's responses to provide additional context and information.
- Testing and Debugging: Extensive testing and debugging were conducted to ensure the smooth functioning of the web application and the chatbot model. User feedback and error logs were analyzed to identify and fix any issues.

#### User Interaction:

- Chat Interface: Users interact with the chatbot through a chat interface embedded in the web application.
- Input Box: Users type messages or queries into the input box provided by the chat interface.
- Real-Time Responses: The web application displays real-time responses generated by the language model and relevant source documents retrieved from the source document retrieval system.

#### UI for my web app
<img src = "./figures/Screenshot 2024-03-21 at 21.06.26.png">

## Acknowledgments:
I would like to express my gratitude to Ma Wut Yee Aung and Rakshya for their support and guidance throughout the project. Special thanks for your contributions to the development and testing of the web application.