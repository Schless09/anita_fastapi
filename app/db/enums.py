from enum import Enum

class CompanyStage(str, Enum):
    PRE_SEED = "Pre-Seed"
    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C_PLUS = "Series C+"
    PRIVATE = "Private"
    PUBLIC = "Public"

class CompanyIndustryVertical(str, Enum):
    AI = "AI"
    HEALTHCARE = "Healthcare"
    FINTECH = "Fintech"
    INSURTECH = "Insurtech"
    ENTERPRISE_SOFTWARE = "Enterprise Software"
    DEVELOPER_TOOLS = "Developer Tools"
    CYBERSECURITY = "Cybersecurity"
    MEDIA = "Media"
    CONSTRUCTION_TECH = "Construction Tech"
    E_COMMERCE = "E-commerce"
    LOGISTICS = "Logistics"
    ROBOTICS = "Robotics"
    CLIMATE = "Climate"
    EDUCATION = "Education"
    LEGAL_TECH = "LegalTech"
    BIOTECH = "Biotech"
    IOT = "IoT"
    CONSUMER = "Consumer"
    REAL_ESTATE = "Real Estate"
    HR_TECH = "HR Tech"
    GAMING = "Gaming"
    TRAVEL = "Travel"
    SUPPLY_CHAIN = "Supply Chain"
    MANUFACTURING = "Manufacturing"

class TargetMarket(str, Enum):
    B2B = "B2B"
    B2C = "B2C"
    ENTERPRISE = "Enterprise"
    SMB = "SMB"
    CONSUMER = "Consumer"

class RoleCategory(str, Enum):
    FRONTEND = "Frontend"
    BACKEND = "Backend"
    FULL_STACK = "Full-Stack"
    INFRA = "Infra"
    DEVOPS = "DevOps"
    DATA = "Data"
    ML_AI = "ML/AI"
    MOBILE = "Mobile"
    SWE = "SWE"
    DESIGN = "Design"
    PRODUCT = "Product"
    SECURITY = "Security"
    FOUNDING_ENGINEER = "Founding Engineer"
    QA = "QA"
    EMBEDDED = "Embedded"

class Seniority(str, Enum):
    ZERO_TO_THREE = "0-3"
    THREE_TO_FIVE = "3-5"
    FIVE_TO_EIGHT = "5-8"
    EIGHT_PLUS = "8+"

class WorkArrangement(str, Enum):
    ON_SITE = "On-site"
    HYBRID = "Hybrid"
    REMOTE = "Remote"

class RoleStatus(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    CLOSED = "Closed"

class AutonomyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

class CodingProficiency(str, Enum):
    BASIC = "Basic"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

class TechBreadth(str, Enum):
    FRONTEND = "Frontend"
    BACKEND = "Backend"
    FULL_STACK = "Full-Stack"
    INFRA = "Infra"
    AI_ML = "AI/ML"

class AIMLExpRequired(str, Enum):
    REQUIRED = "Required"
    PREFERRED = "Preferred"
    NOT_REQUIRED = "Not Required"

class ProductStage(str, Enum):
    PROTOTYPE = "Prototype"
    MVP = "MVP"
    MARKET_READY = "Market-ready"
    SCALING = "Scaling"
    ESTABLISHED = "Established"

class DevMethodology(str, Enum):
    AGILE = "Agile"
    SCRUM = "Scrum"
    KANBAN = "Kanban"
    HACKER = "Hacker"
    OWNER = "Owner"
    N_A = "n/a"

class EducationRequired(str, Enum):
    TOP_30_CS = "Top 30 CS program"
    IVY_LEAGUE = "Ivy League"
    RESPECTED_SCHOOLS = "Respected schools"
    NO_REQUIREMENT = "No requirement"
    NO_BOOTCAMPS = "No bootcamps"

class EducationAdvancedDegree(str, Enum):
    PHD_PREFERRED = "PhD preferred"
    MASTERS_PREFERRED = "Master's preferred"
    NOT_REQUIRED = "Not required" 