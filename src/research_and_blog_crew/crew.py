from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai import LLM
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


groq_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0.2
)

#define the class for our crew
@CrewBase
class ResearchAndBlogCrew():
    """ResearchAndBlogCrew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    #define the paths of config files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    #========Agents========
    @agent 
    def report_generator(self)-> Agent:
        return Agent(config=self.agents_config["report_generator"], llm=groq_llm)    
    
    @agent 
    def blog_writer(self)-> Agent:
        return Agent(config=self.agents_config["blog_writer"], llm=groq_llm)      
    
    #========Tasks========
    #order of task definition matters
    @task
    def report_task(self)-> Task:
        return Task(config=self.tasks_config["report_task"])

    def blog_writer_task(self)-> Task:
        return Task(config=self.tasks_config["blog_writer_task"],
            output_files = "blogs/blog.md"
        )
    
    
    #==========Crew=========
    @crew 
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process = Process.sequential,
            verbose=True
        )
