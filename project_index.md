# Portfolio

<div align="center">

![Portfolio Banner](https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=500&size=40&pause=1000&color=58A6FF&center=true&vCenter=true&width=600&height=100&lines=AI+%26+ML+Projects)

</div>

<div style="display: flex; justify-content: space-between; align-items: center; margin: 20px 0;">
    <h2 style="margin: 0;">Projects</h2>
    <select id="tagFilter" onchange="filterProjects()" style="padding: 4px 8px; border-radius: 4px; border: 1px solid #30363d; font-size: 14px; color: #c9d1d9; background: #0d1117;">
        <option value="all" selected>All Projects</option>
        <option value="AI">AI</option>
        <option value="LLM">LLM</option>
        <option value="Gradio">Gradio</option>
        <option value="Python">Python</option>
        <option value="Customer Service">Customer Service</option>
    </select>
</div>

<script>
function filterProjects() {
    const filter = document.getElementById('tagFilter').value;
    const projects = document.getElementsByClassName('project-row');
    
    for (let project of projects) {
        if (filter === 'all' || project.getAttribute('data-tags').includes(filter)) {
            project.style.display = '';
        } else {
            project.style.display = 'none';
        }
    }
}
</script>

<div align="center">

| Project | Description | Links | Tags |
|:--------|:------------|:-----:|:----:|
| ✈️ Airline AI Assistant | AI-powered airline customer service assistant | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/anuj-kumar-30/LLM-engieering) [![Demo](https://img.shields.io/badge/Demo-238636?style=flat-square&logo=gradio&logoColor=white)](https://cea988592e0df1c2fb.gradio.live) | <div class="project-row" data-tags="AI,LLM,Gradio,Python,Customer Service"><img src="https://img.shields.io/badge/AI-58A6FF?style=flat-square&logo=openai&logoColor=white" alt="AI"/> <img src="https://img.shields.io/badge/LLM-58A6FF?style=flat-square&logo=openai&logoColor=white" alt="LLM"/> <img src="https://img.shields.io/badge/Gradio-238636?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/> <img src="https://img.shields.io/badge/Python-58A6FF?style=flat-square&logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/Customer_Service-58A6FF?style=flat-square&logo=service&logoColor=white" alt="Customer Service"/></div> |

</div>

## Tags

<div align="center">

[![AI](https://img.shields.io/badge/AI-58A6FF?style=flat-square&logo=openai&logoColor=white)](#ai)
[![LLM](https://img.shields.io/badge/LLM-58A6FF?style=flat-square&logo=openai&logoColor=white)](#llm)
[![Gradio](https://img.shields.io/badge/Gradio-238636?style=flat-square&logo=gradio&logoColor=white)](#gradio)
[![Python](https://img.shields.io/badge/Python-58A6FF?style=flat-square&logo=python&logoColor=white)](#python)
[![Customer Service](https://img.shields.io/badge/Customer_Service-58A6FF?style=flat-square&logo=service&logoColor=white)](#customer-service)

</div>

## Project Details

### Airline AI Assistant
- **Description**: AI-powered airline customer service assistant using Gemini API
- **Tech Stack**: Python, Gradio, Gemini API
- **Features**:
  - Real-time customer service responses
  - Natural language understanding
  - User-friendly interface
  - Concise responses
- **Links**:
  - [GitHub](https://github.com/anuj-kumar-30/LLM-engieering)
  - [Live Demo](https://cea988592e0df1c2fb.gradio.live)

---

<div align="center">

![Last Updated](https://img.shields.io/badge/Updated-2024--03--19-30363d)

</div>

<style>
:root {
    color-scheme: dark;
}

body {
    background-color: #0d1117;
    color: #c9d1d9;
}

select {
    font-size: 14px;
    font-weight: 400;
    color: #c9d1d9;
    background-color: #0d1117;
    cursor: pointer;
    transition: all 0.2s ease;
}

select:hover {
    border-color: #58A6FF;
}

.project-row {
    transition: all 0.2s ease;
}

.project-row:hover {
    transform: scale(1.01);
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    color: #c9d1d9;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #30363d;
}

th {
    font-weight: 500;
    color: #8b949e;
}

tr:hover {
    background-color: #161b22;
}

a {
    color: #58A6FF;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

h1, h2, h3 {
    color: #c9d1d9;
}

</style> 