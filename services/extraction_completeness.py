"""Conservative source-aware Sarvam checks; never infer or add extracted facts."""
import re

from models.schemas import JobDescriptionData, ResumeData

TECH_FIELDS = ('skills', 'tools', 'programming_languages', 'frameworks', 'databases', 'cloud_platforms')
JD_TECH_FIELDS = ('required_skills', 'required_tools', 'preferred_skills', 'preferred_tools')
SECTION = re.compile(r'(?im)^\s*(?:[#*•-]+\s*)?(?:technical skills|skills|requirements|'
                     r'qualifications|responsibilities|experience|employment|work history|'
                     r'education|projects|achievements)\s*(?::|$)')
SKILL_SECTION = re.compile(r'(?im)^\s*(?:[#*•-]+\s*)?(?:(?:required|preferred|technical)\s+)?'
                           r'(?:skills|technologies|tools|tech stack)\s*(?::|\n)\s*\S')
TECH_MENTION = re.compile(r'\b(?:python|javascript|typescript|java|react|angular|sql|postgresql|'
                          r'mongodb|aws|azure|docker|kubernetes|git|linux|c\+\+|c#|node\.js)\b', re.I)


def completeness_problem(data: JobDescriptionData | ResumeData, text: str) -> str | None:
    """Return a fixed reason for clear omissions, not a completeness percentage.

    Substantial means >=80 words, >=400 characters, or >=30 words with at least
    two section headings. Optional sections are never required just by schema.
    These checks cannot establish that every fact in a document was extracted.
    """
    words = len(text.split())
    substantial = words >= 80 or len(text.strip()) >= 400 or (words >= 30 and len(SECTION.findall(text)) >= 2)
    if not substantial:
        return None
    if isinstance(data, JobDescriptionData):
        technologies = any(getattr(data, name) for name in JD_TECH_FIELDS)
        usable = technologies or data.minimum_experience_years or data.maximum_experience_years or (
            data.education_requirements or data.responsibilities or data.certifications or data.domain)
        if not usable:
            return 'The substantial job description yielded no usable requirements or responsibilities.'
        requirements = re.search(r'\b(?:requirements|required|must have|qualifications|preferred)\b', text, re.I)
        if not technologies and (SKILL_SECTION.search(text) or (requirements and TECH_MENTION.search(text))):
            return 'The job description contains skill or technology requirements but none were extracted.'
    else:
        technologies = any(getattr(data, name) for name in TECH_FIELDS)
        qualifications = technologies or data.roles or data.responsibilities or data.education or (
            data.projects or data.achievements or data.total_experience_years or data.certifications)
        if not qualifications:
            return 'The substantial resume yielded no candidate qualifications.'
        if not technologies and SKILL_SECTION.search(text):
            return 'The resume contains a skills section but no skills or technologies were extracted.'
    return None


def sarvam_instructions(schema) -> str:
    common = '''Inspect the ENTIRE supplied document, including every section and bullet.
Map all document-supported facts into the existing JSON schema, with verbatim evidence.
Use empty lists/null only when the source genuinely does not support that field.
Do not invent facts, infer unstated qualifications, or calculate an ATS score.'''
    if schema is JobDescriptionData:
        return common + '''\nCover role/title, required skills, preferred skills, required tools/technologies,
preferred tools/technologies, experience requirements, education requirements,
responsibilities and evidence. A title and responsibilities alone are insufficient
when the source also lists skills or qualifications. Preserve required/preferred distinctions.'''
    return common + '''\nCover candidate-supported skills, programming languages, frameworks, tools,
databases, cloud platforms, positions and dates, responsibilities/bullets, education,
projects, achievements, explicitly stated experience and evidence. Contact information
alone is insufficient when the source also contains qualifications or work history.'''


CORRECTIVE_INSTRUCTION = '''The previous structured response omitted substantial source information.
Re-read the entire original document and return a complete replacement using the SAME schema.
Check each supported qualification and requirement section. Do not invent missing facts.'''
