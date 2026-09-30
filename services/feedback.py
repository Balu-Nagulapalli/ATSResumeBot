"""Feedback is derived only from matching results; it never invents candidate history."""
from models.schemas import MatchResult

DOCKER_TOPICS = ['Containers and images', 'Dockerfile', 'Networking and volumes',
                 'Docker Compose', 'Deploy a small application']


def generate_feedback(result: MatchResult) -> dict:
    criteria = result.criteria
    strengths = [f'{name.replace("_", " ").title()}: {item.score:.0f}% evidence coverage.'
                 for name, item in criteria.items() if item.applicable and item.score >= 85]
    required = criteria['required_skills'].missing_requirements
    preferred = criteria['preferred_skills'].missing_requirements
    tools = criteria['tools'].missing_requirements
    required_tools = [m.requirement for m in criteria['tools'].matches if m.required and m.credit < 1]
    preferred_tools = [m.requirement for m in criteria['tools'].matches if not m.required and m.credit < 1]
    improvements = [f'If you have experience with {skill}, mention it explicitly with a concrete example.'
                    for skill in required + tools]
    improvements += [f'Clarify {item.lower()} using truthful, relevant details.'
                     for item in criteria['resume_quality'].missing_requirements]
    roadmap = []
    for skill in list(dict.fromkeys(required + required_tools + preferred + preferred_tools))[:8]:
        roadmap.append({'topic': skill, 'priority': 'High' if skill in required + required_tools else 'Optional',
            'why': 'Requested by this role but not fully evidenced in the resume.',
            'learn': DOCKER_TOPICS if skill.lower() == 'docker' else [
                f'Learn the fundamentals of {skill} from its official documentation.',
                f'Build a small project that demonstrates {skill}.',
                'Test the project and document the decisions and results.',
                'Describe your work accurately once you have completed it.']})
    return {'strengths': strengths, 'missing_required_skills': required,
            'missing_preferred_skills': preferred, 'tool_gaps': tools,
            'experience_gaps': criteria['experience'].missing_requirements,
            'education_gaps': criteria['education'].missing_requirements,
            'certification_gaps': criteria['certifications'].missing_requirements,
            'responsibility_gaps': criteria['responsibilities'].missing_requirements,
            'improvements': improvements, 'learning_roadmap': roadmap}
