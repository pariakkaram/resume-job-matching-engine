# Label mapping review

Generated 2026-10-25 18:15 UTC by `scripts/review_label_mapping.py` from `configs/labels.yaml` (status: proposed) and the files in `data/raw/`. Every number below was computed from the data; regenerate after editing the config.

- Jobs after preview dedup: 14070; with at least one family: 10630; with several: 1734.
- Categories with fewer than 20 grade-2 jobs: AGRICULTURE (excluded) (10).
- Categories in the data but missing from the config: none.

## 1. Resume categories (from the data)

| Category | Resumes | Status | Primary families | Related families |
|---|---|---|---|---|
| ACCOUNTANT | 118 | included | accounting | finance, banking |
| ADVOCATE | 118 | included | social_services_advocacy | customer_service |
| AGRICULTURE | 63 | excluded | agriculture | science_research |
| APPAREL | 97 | included | apparel_fashion | retail, design |
| ARTS | 103 | included | arts | teaching, design, digital_media |
| AUTOMOBILE | 36 | included | automotive |  |
| AVIATION | 117 | included | aviation | engineering |
| BANKING | 115 | included | banking | finance, accounting |
| BPO | 22 | included | customer_service | operations_management |
| BUSINESS-DEVELOPMENT | 120 | included | business_development | sales, marketing, consulting |
| CHEF | 118 | included | culinary | hospitality |
| CONSTRUCTION | 112 | included | construction | engineering, project_management |
| CONSULTANT | 115 | included | consulting | project_management |
| DESIGNER | 107 | included | design | arts, digital_media |
| DIGITAL-MEDIA | 96 | included | digital_media | marketing, public_relations, design |
| ENGINEERING | 118 | included | engineering | software_it, project_management, manufacturing_production |
| FINANCE | 118 | included | finance | accounting, banking |
| FITNESS | 117 | included | fitness |  |
| HEALTHCARE | 115 | included | healthcare | social_services_advocacy |
| HR | 110 | included | hr |  |
| INFORMATION-TECHNOLOGY | 120 | included | software_it | data_analytics, project_management |
| PUBLIC-RELATIONS | 111 | included | public_relations | marketing, digital_media |
| SALES | 116 | included | sales | retail, business_development, customer_service |
| TEACHER | 102 | included | teaching |  |


## 2. Random sample of 36 job titles (seed 0)

Store Receiver - Part Time [-]; Director of eCommerce Website Product Management & Optimization [digital_media]; SailPoint IDN Engineer (Contract) [-]; Certified Caregiver [healthcare]; Manager, Technical Accounting [accounting]; Senior UX Researcher, Pixel [design]; Relationship Banker - Stamford Downtown Financial Center - Stamford, CT [finance, banking]; Director of Wire Harness Engineering (Automotive) [automotive]; Director Project Finance [finance]; Junior Groomer [-]; Data Engineer [data_analytics]; Assistant Store Manager [retail]; Validation Engineer (Onsite) [engineering]; Director of FP&A [finance]; Staff Accountant [accounting]; IT Support Engineer I, IT Services [software_it]; Day Porter Lead [-]; Network Administrator [software_it]; Janitorial Worker [-]; Principal Data Architect [data_analytics]; Senior Information Security Analyst [software_it]; Lead Software Engineer [software_it]; Product Owner III (Contractor) [project_management]; Assistant Manager(07462) - 9310 S Eastern Ave [-]; Associate Attorney [legal]; Mental Health Clinician [healthcare]; Project Manager [project_management]; OTR Driver - Kansas City, Kansas [logistics_warehouse]; Travel MedSurg RN in Durham, NC - Make $2221.40/week [healthcare]; Linux Engineer [software_it]; Dispatcher [logistics_warehouse]; Junior Project Developer [software_it]; IT Service Desk Manager [software_it]; Luxury Sales Professional [sales]; Lead Systems Engineer [software_it, engineering]; Senior Program Protection Planning and Security Support [software_it]

## 3. Job families

10630 of 14070 jobs (75.6%) match at least one family; 1734 (12.3%) match more than one.

| Family | Jobs | Corroborating codes | Most common codes | Used by |
|---|---|---|---|---|
| software_it | 1583 | IT | IT 79%, ENG 35%, SALE 3%, OTHR 3% | ENGINEERING (R), INFORMATION-TECHNOLOGY (P) |
| data_analytics | 355 | ANLS, IT | IT 66%, ENG 26%, ANLS 14%, RSCH 6% | INFORMATION-TECHNOLOGY (R) |
| engineering | 621 | ENG, QA | ENG 72%, IT 52%, MNFC 14%, MGMT 9% | AVIATION (R), CONSTRUCTION (R), ENGINEERING (P) |
| healthcare | 1474 | HCPR | HCPR 70%, OTHR 11%, IT 5%, RSCH 4% | HEALTHCARE (P) |
| sales | 1030 | BD, SALE | SALE 83%, BD 67%, OTHR 6%, CUST 4% | BUSINESS-DEVELOPMENT (R), SALES (P) |
| business_development | 135 | BD, SALE | BD 69%, SALE 69%, MRKT 9%, CNSL 6% | BUSINESS-DEVELOPMENT (P), SALES (R) |
| customer_service | 283 | CUST | OTHR 43%, CUST 16%, SALE 12%, MGMT 6% | ADVOCATE (R), BPO (P), SALES (R) |
| accounting | 719 | ACCT, FIN | ACCT 81%, FIN 66%, HR 4%, ADM 4% | ACCOUNTANT (P), BANKING (R), FINANCE (R) |
| finance | 488 | ACCT, FIN | FIN 72%, SALE 37%, ACCT 30%, IT 8% | ACCOUNTANT (R), BANKING (R), FINANCE (P) |
| banking | 186 | FIN | FIN 61%, SALE 57%, BD 10%, MGMT 9% | ACCOUNTANT (R), BANKING (P), FINANCE (R) |
| hr | 350 | HR | HR 74%, SALE 9%, IT 7%, BD 7% | HR (P) |
| consulting | 270 | CNSL | IT 39%, SALE 33%, CNSL 18%, BD 11% | BUSINESS-DEVELOPMENT (R), CONSULTANT (P) |
| construction | 334 | (none) | MGMT 55%, MNFC 51%, PRJM 17%, OTHR 13% | CONSTRUCTION (P) |
| marketing | 284 | ADVR, MRKT, PR | MRKT 69%, SALE 54%, BD 9%, IT 6% | BUSINESS-DEVELOPMENT (R), DIGITAL-MEDIA (R), PUBLIC-RELATIONS (R) |
| digital_media | 211 | ADVR, ART, MRKT, PR, WRT | MRKT 64%, SALE 34%, WRT 25%, PR 21% | ARTS (R), DESIGNER (R), DIGITAL-MEDIA (P), PUBLIC-RELATIONS (R) |
| public_relations | 65 | MRKT, PR, WRT | MRKT 69%, PR 54%, WRT 43%, SALE 17% | DIGITAL-MEDIA (R), PUBLIC-RELATIONS (P) |
| legal | 222 | LGL | LGL 86%, ADM 5%, GENB 3%, CNSL 3% | unmapped (negatives only) |
| social_services_advocacy | 116 | (none) | OTHR 55%, HCPR 22%, CUST 6%, MGMT 5% | ADVOCATE (P), HEALTHCARE (R) |
| automotive | 140 | (none) | MNFC 54%, MGMT 51%, ENG 16%, OTHR 12% | AUTOMOBILE (P) |
| design | 208 | ART, DSGN | DSGN 62%, IT 58%, ART 53%, ENG 17% | APPAREL (R), ARTS (R), DESIGNER (P), DIGITAL-MEDIA (R) |
| teaching | 201 | EDU, TRNG | EDU 64%, TRNG 50%, OTHR 11%, HCPR 9% | ARTS (R), TEACHER (P) |
| culinary | 133 | (none) | MGMT 85%, MNFC 73%, OTHR 8%, GENB 5% | CHEF (P) |
| hospitality | 195 | (none) | MGMT 75%, MNFC 70%, OTHR 9%, IT 8% | CHEF (R) |
| fitness | 25 | (none) | OTHR 52%, MGMT 16%, HCPR 16%, SALE 12% | FITNESS (P) |
| aviation | 49 | (none) | IT 41%, ENG 33%, MNFC 31%, MGMT 27% | AVIATION (P) |
| agriculture | 10 | (none) | DSGN 30%, IT 20%, RSCH 20%, CNSL 20% | unmapped (negatives only) |
| apparel_fashion | 108 | (none) | SALE 58%, MRKT 30%, MGMT 14%, IT 13% | APPAREL (P) |
| retail | 333 | CUST, SALE | SALE 68%, BD 43%, MGMT 14%, OTHR 14% | APPAREL (R), SALES (R) |
| arts | 50 | ART, DSGN | ART 48%, DSGN 40%, IT 38%, EDU 16% | ARTS (P), DESIGNER (R) |
| admin_office | 301 | ADM | ADM 78%, HCPR 5%, OTHR 4%, LGL 4% | unmapped (negatives only) |
| logistics_warehouse | 651 | DIST, MNFC, PRCH, SUPL | MGMT 62%, MNFC 62%, SUPL 14%, PRCH 10% | unmapped (negatives only) |
| manufacturing_production | 154 | MNFC, PROD, QA | MNFC 58%, MGMT 47%, QA 14%, MRKT 14% | ENGINEERING (R) |
| project_management | 538 | MGMT, PRDM, PRJM | PRJM 59%, IT 54%, PRDM 14%, ENG 11% | CONSTRUCTION (R), CONSULTANT (R), ENGINEERING (R), INFORMATION-TECHNOLOGY (R) |
| science_research | 284 | RSCH, SCI | RSCH 61%, IT 57%, ANLS 50%, SCI 12% | unmapped (negatives only) |
| operations_management | 189 | GENB, MGMT | MGMT 68%, MNFC 55%, SALE 7%, GENB 6% | BPO (R) |
| security_public_safety | 21 | (none) | MGMT 81%, IT 71%, OTHR 62%, MNFC 19% | unmapped (negatives only) |
| real_estate_insurance | 176 | (none) | SALE 61%, MGMT 36%, FIN 22%, BD 16% | unmapped (negatives only) |

### Sample titles per family

- **software_it**: Computer Science Curriculum Designer, Alpha - an EDU Startup (Remote) - $100,000/year USD; Intern - DRAM Cloud Engineer; Quality Application Engineer; Frontend Web Developer (Machine Learning); Senior Computer Network Support Specialist; Kitting & Packaging Coordinator, SAP; Industrial Programmer; Senior Java Fullstack developer; Infrastructure Architect; Unreal Engine 4 Programmer; Java Consultant; System Integration Engineer
- **data_analytics**: Data Analytics Consultant; Manager II, Machine Learning; Manager, Audience Analytics; Senior Data Scientist - Algorithms; Data architect/Data analyst; Data Scientist; Data Analyst; Data Analyst; Contractor-Laboratory Capabilities, Clinical Data Management (Consultant); AI District 23 - Store Hourly; Data Governance Specialist; Data Scientist / Operations Research Analyst
- **engineering**: Imaging Engineering Specialist II; Senior Test Engineer; Quality Engineer III; Senior Director of Engineering; Water Process Engineer; Associate Test Engineer; On-Site Project Engineer; DWDM Test Engineer; Senior Civil Engineer; Chief Systems Engineer; Systems Engineer; Mechanical Engineer
- **healthcare**: Pharmaceutical Sales Specialist - R&I Primary Care (Raleigh NC); Surgery Scheduler; CNA - On-Call; Physician Assistant or Nurse Practitioner (APRN) - Clinical Lead / Cardiology Norwich Salem; Medical Collector (Major Medical Benefit Claims); Physical Therapist - Eligible Orthopaedic Resident; Inpatient Therapy Technician (CNA), Per Diem Day Shift, Inpatient Rehabilitation; Radiology Technologist; Surgical Technologist; Director of Nursing; Psychiatrist Needed for Locums Coverage at Facility in Dallas, Texas; Optometrist
- **sales**: Commodity Sales & Sourcing Specialist- Remote; Hospitality Sales & Event Specialist; Associate Director, Commerce Media (Sales); Sales Manager; Business Development Manager - Outside Sales; Salesperson; Account Executive; Store Sales Associate; REMOTE STEEL BUILDING SALES MAKE $1,000 TO $3000 TO $10,000 PER ORDER!; Bilingual Account Manager (Mandarin); Regional Sales Manager, UK; District Sales Manager, Oklahoma
- **business_development**: Director, Business Development (West Coast); Associate Director of Corporate Partnership Relations; Business Development Representative; Director of Business Development; Food Production Manager, New Business Development; Business Development Manager; Business Development Manager; Director of Business Development; Business Development Manager; DEVELOPMENT AND PARTNERSHIPS INTERN; Business Development Representative; Director of Business Development
- **customer_service**: Client Care Center Specialty Care- Spanish Care (English/Spanish); Customer Service Consultant; Principal Program Manager, Customer Service by Amazon; Client Service Partner, Conferences; Customer Success Director; Client Service Associate*; Supervisor, IS Client Support; Customer Service Representative; Enterprise, Strategic Customer Success Manager; Call Center Representative; Customer Service Representative (Remote Possible); Client Services Associate
- **accounting**: Senior Accountant - Record to Report; Staff Accountant; Audit Intern - Winter 2025; Accounting Manager ; Billing Representative II (A/R) - Central Billing Office *Onsite - Las Vegas, NV*; Payroll Coordinator- Available ASAP; Controller; Accounts Payable Processor; Senior Manger General &amp; Technical Accounting - B&amp;T; Plant Accountant; Sr Accounting Manager; Audit Senior Associate | Investment Companies
- **finance**: Financial Services Representative; SAP Central Finance Technical; Quantum FIS Treasury; Financial Advisor - Entry Level (2024); Controller; IPV/Valuations Manager, Investment Bank; Relationship Banker - Douglas Centre Financial Center - Coral Gables, FL - Spanish Language Required; Administrative and Financial Coordinator; Senior Financial Analyst; Oracle Finance Functional Consultant; Financial Crimes Investigator Sr - GA AL TN SC FL - Remote; Research And Administrative Assistant (Hedge Fund)
- **banking**: HOA Banking Team Leads, Relationship Managers, and Client Solutions Managers - Southern California; Teller / Customer Service Rep; Bank Employee; Underwriting Assistant; Associate Underwriter; Loan Officer; Branch Manager; Universal Banker - San Leandro; Universal Tellers & Relationship Bankers (West Valley - South); Loan Officer; Commercial Lending Financial Analyst; Restoration Branch Manager
- **hr**: Recruitment Consultant; Payroll Manager; Compensation Consulting - Engagement Leader; Client Manager - Business Development Recruitment; Partnership Coordinator and Student Recruiter; Human Resources Director; Human Resources Assistant; Staffing Recruiter; Craft Employee Relations Manager; Payroll Accountant; HR ServiceNow Product Owner; Customer Advocate – Payroll
- **consulting**: Robotic Process Automation Consultant; Commercial Sales Consultant; Sales Consultant - Steamboat Springs, CO; Compliance Consultant; Business Analytics Consultant- Revenue Cycle; Leasing Consultant;  PHD Admission Essay Consultant (Statistics,Cancer,Cell,Biostas,Gene,Bio-medical-related,Psychology); Design Consultant; Oracle Functional Consultant; OMP Consultant ; Sales Consultant; Technical Consultant - zOS (TS/SCI)
- **construction**: Supervisor Network Construction; Construction Cost Estimator; Shipyard Estimator and Contracts Administrator ; HVAC Truck Based Sr Controls Technician; Construction Manager; General Laborer; Construction Project Manager; Desk Superintendent, Rail Control Center; Track Laborer - Bement & Tilton, IL; Industrial Electrician; Estimator; Site Superintendent
- **marketing**: Social Media and Email Marketing Manager; Sales And Marketing Intern; Marketing Manager; Events Specialist; VP, Channel Marketing; Product Marketing Manager; Marketing Manager; Marketing Generalist; Affiliate Marketing Manager; Advertising Account Executive - Digital; Event Coordinator; Junior Marketing Manager
- **digital_media**: Manager, Digital Content Producer | Global Beauty Company; Podcast Social Media Marketing & Digital Video Editing Intern; Senior Associate, Programmatic; Retirement Plan Producer; Social Media Specialist; Copywriter; NEWS PRODUCER - WSAZ; Senior Digital Marketing Manager, Paid Search; Content Manager; Social Media Manager; Web Content Developer; Publisher, Local Marketing Specialist & Parent Influencer
- **public_relations**: Internal Communications Specialist; Marketing And Public Relations Intern; Public Relations Account Director: Emerging Technology & Consumer Finance; Corporate Communications Manager; Logistics Communications Manager, Sr. - LAX Airport; Head of Communications; Digital Design Communication Specialist; Communications Manager; Sr. Media Relations Specialist; Senior Marketing Communications Manager; MARKETING AND COMMUNICATIONS MANAGER; Public Relations Account Executive
- **legal**: Paralegal; Legal Secretary; Paralegal; Associate Attorney (New Orleans); Senior Litigation Attorney/Special Assistant Attorney General/Government Litigation #13867; Employment Litigation Associate; Paralegal; Business & Trademark Attorney; Board and Legal Specialist; Real Estate Paralegal; Attorney, Self-Help Services; Legal Billing Specialist
- **social_services_advocacy**: Lead Housing Coordinator; Industrial Base Outreach Analyst; Family Services Intake Clinician- Part Time; Case Worker, Crisis Intervention; Remote Licensed Clinical Social Worker - Ohio; Registered Nurse Case Manager; Case Manager Assistant; Educational Assistant, Functional Life Skills; Residential Youth Specialist; Customer Advocate; Outreach and Recruiting Specialist; Social Services Attorney
- **automotive**: Investment Banking Vice President | Automotive M&A; Diesel Technician/Mechanic III - Entry Level; Diesel Mechanic Position Now Open!!!!!; Fleet & Equipment Specialist; NO 1 Auto Machine Mechanic; Automotive Installer; Automotive Parts Specialist; Design Engineer- Automotive Parts manufacturer (33778); SALES MANAGER - (Automotive Tier 1); (Night Shift) Maritime Diesel Engine Analyst; Lead Diesel Technician; Dealership Operations Manager
- **design**: Head of Visual Design Operating Systems; Creative Designer; UI Web Architect; Interior Designer Project Manager; Senior Product Designer; UI/UX Designer; UI Angular Developer; Mid-Level Civil BIM Designer; Lead Landscape Designer/Architect; Junior Art Director; Digital Designer; EWP Drafter Truss Design
- **teaching**: Mathematics Teacher; Associate Teacher; Salmon Bay K-8 - Special Education Teacher - 0.5 FTE; Academic Coach, Trilogy (Remote) - $30,000/year USD; High School Social Studies Teacher; Wellness Instructor; Middle School English Language Arts Teacher; Classroom Teacher; Lead Preschool Teacher; Assistant Professor; Seasonal School Events Associate; Faculty Group Practice Ultrasound Technologist - **Per Diem - Obs/Gyn/MFM experience highly preferred**
- **culinary**: Cook; Cook III; Line Cook; Food Service Worker; GUCCI Pastry Cook Osteria da Massimo Bottura - Beverly Hills; Restaurant Crewmember - Cook, Cashier, and Customer Service; Dishwasher; Bakery Specialist - JVSS10 - Fry/Clay - Part-Time; Dishwasher; Kitchen Supervisor; Baker FT; Kitchen Aide  
- **hospitality**: In-Room Dining Server; Restaurant Manager - DraftKings Sports & Social Troy, MI; Director of Food And Beverage; Hourly Restaurant Manager; Bartender - Union Tavern; LTC Sr. Restaurant Associate (PDI); Host for the Restaurant + Bar; Restaurant Manager; Waiter/Server; Food Runner; Hospitality Concierge - Womens Health *Per Diem*; Assistant General Manager, MTC Dining - 126042
- **fitness**: Wellness Coach; Fitness Studio Manager | Culver City, CA; Permanent Staff Athletic Trainer job in Brunswick, GA - Make $21/hour; Exercise Physiologist; Fitness Trainer; Fitness Trainer; Customer Service Representative, Fitness (Remote, $15/hour); Coach Swim; Fitness Manager; Athletic Trainer - Per Diem; Personal Trainer; Group Fitness Instructor
- **aviation**: Mechanical Design Engineer (Aerospace Exp.); Aerospace VP of Sales and Marketing; Office Engineer II - Airport Operational Readiness - Training Coordinator - LAX Airport; Airport EOC Network Administrator (D); Airport Manager- Insider Threat Program; Operations Manager (STL Airport); Avionics Systems Engineer; Helicopter Float Pilot - HAA; Challenger 604 First Officer; Aerospace Structural Design Engineer; Pilot; Airport Manager- Emergency Management Trainer
- **agriculture**: Senior Crop Specialist; Forester - Multiple Vacancies; Lead Landscape Designer/Architect; Grower Enrollment Specialist; Forestry Technician/ Inventory Forester; Landscape Designer; Landscape Architect; Commercial Landscape Superintendent; VP of Grain; Landscape Manager at Refined Gardens
- **apparel_fashion**: Apparel Sales Outfitter; Marketing Merchandising Manager; Full-time Nabisco Sales Service Representative/Merchandiser Order Writer; Beauty Advisor; Product Location Merchandiser; Seasonal Retail Sales Associate, Northridge Fashion Center; Apparel Sales Representative - Clarksville & Hendersonville Tennessee; Merchandise and Service Coordinator; Part-Time Beauty Advisor - Sephora; Retail Merchandise Processor - PT; Anthropologie Weddings Stylist Part-Time; Retail Sales Associate Footwear
- **retail**: Retail Associate; Assistant Deli Manager; Retail Store Manager; Store Manager (Boston Area); 10161 - Retail Cashier; Cashier & Front End Services; Urban Outfitters Key Holder Part-Time; Director of Retail Operations, Full time (Oberlin College) $65k/year!; Restaurant Crewmember - Cook, Cashier, and Customer Service; Free People Key Holder Part-Time; Retail Sales Associate, Santa Rosa Plaza - Part Time; Seasonal Retail Sales Associate - Footwear
- **arts**: Senior Illustrator; Part-Time Lecturer in Arts Administration Program, Metropolitan College, Arts Administration; Head Strings Music Teacher - 0.5 FTE; Art Teacher; Corporate Sponsorship Specialist for Nationwide Art Battle Events & TV Show; Language Arts Instructor - Part Time; Senior Gameplay Animator - XDefiant; Public Art Manager; Personal Assistant (High Profile Artist); Creative Arts Therapist; Regional Managing Director, Post War & Contemporary Art; 3D Artist
- **admin_office**: Administrative Technician; Administrative Assistant; Unit Secretary MV NICU; Secretary Patient Care - EVENING SHIFT - Medical Intermediate Care; Administrative Accounting Assistant; Office Manager; Contract Administrative Assistant; Receptionist-Part Time; Branch Office Administrator - Chisago City, MN; Administrative Assistant; Executive Assistant - Legal; Executive Assistant
- **logistics_warehouse**: Executive Project Director, Transit Development and Delivery; Loader; Truck Driver; Forklift Operator III; Inventory Control Assistant; LOCAL CDLA Feed Haul Truck Driver; Warehouse Worker; Warehouse Receiving Supervisor; MRO - BUYER; Tow Truck Driver 2-Car; Delivery Truck Driver - CDL A; Bulk Rack Loader
- **manufacturing_production**: Production Associate (743922); Production Operator (2nd Shift); Manufacturing Operator 1; Senior Quality Inspector; Production Associate (743950); Quality Technician; Manufacturing Supervisor ; CNC Machinist Trainee; Assembler I (5:00am to 1:30pm); Manufacturing Operator; Battery Production Technician; Fabricator - 1st Shift
- **project_management**: Project Manager; IT Project Coordinator - JIRA; Sr. Product Manager; Senior Projects Control Engineer, Program Manager III; Program Manager; Project Manager – Brownfields and/or Federal Projects; Project Manager; Project Manager; Construction Project Manager; Program Manager; Floor Coatings salesperson and project manager; Project Manager
- **science_research**: R&D Scientist; Lab Manager / Sr. Network Specialist; Senior Data Scientist; Research Project Coordinator - Part-Time 2 year contract; Research Projects Manager - Civil and Environmental Engineering; Clinical Research Manager; Senior Clinical Research Associate - Buffalo/Syracuse, NY (Remote-Regional); Laboratory Technician; Data Scientist – (Remote); Scientist; Senior Data Scientist; Staff Scientist, Catalyst Chemist
- **operations_management**: Operations Manager; General Manager Strategy and Operations – Texas; Operations Coordinator - $20/hr-$22/hr - Washington, DC; Sales Operations Manager (Americas); Operations Manager; General Manager- Head of Consumer Products; Operations Analyst; General Manager; Director of Operations; Operations Manager - Victory Packaging; Area Director of Operations; Senior Marketplace Operations Specialist
- **security_public_safety**: Security Officer | Overnight ; Police Validations Specialist; Business Information Security Officer - GCS; Security Officer; Security Officer, Evening Shift, Security Services; Security Officer I; Security Officer 2 - Taylor; Public Safety Applications Sales Representative (FL/GA); Asset Protection / Loss Prevention (LP) Investigator / Associate - Saks Fifth Avenue; Armed Maritime Security Officer; Police Services Specialist; Part-Time Loss Prevention Associate
- **real_estate_insurance**: Insurance Defense Attorney - 100% Remote; Real Estate Investment Analyst; Assistant Property Manager - $1,500 Sign-On Bonus; Assistant Property Manager; Property Field Claims Adjuster (Fox Valley/Green Bay, WI); Assistant Property Manager, Multifamily; Real Estate Paralegal; Multifamily Real Estate Analyst; Assistant Property Manager; Revenue Cycle Claims Specialist-Temp; Claims Specialist; Relief Property Manager

### Most frequent titles with no family (3440 jobs)

maintenance technician (22); business analyst (21); executive director (10); technical writer (9); service technician (9); team member (8); tech 1, installation (8); engineer (7); senior business analyst (7); maintenance supervisor (7); assistant manager (6); facilities manager (6); field service technician (6); manager (6); board certified behavior analyst (bcba) (6); lead service associate (6); field supervisor (5); product analyst (5); transportation manager (5); hair stylist (5); production planner (5); custodian (5); associate (5); quality assurance analyst (5); environment, health and safety manager (4); architect (4); program coordinator (4); operator (4); district manager (4); field technician (4); maintenance manager (4); quality supervisor (4); chief of staff (4); compliance specialist (4); intern (4); installation technician (4); junior groomer (4); porter (4); veterinary assistant (4); quality manager (4); operations intern-summer 2024 (4); member advisor (4); equipment technician (4); office associate (3); grants manager (3); area manager (3); automation technician (3); project assistant (3); technician (3); project control specialist (3); associate veterinarian (3); senior analyst (3); product development manager (3); chief development officer (3); assistant (3); quality assurance engineer (3); collections specialist (3); support engineer (3); operations support specialist (3); industrial maintenance technician (3)

### Most common multi-family combinations

accounting + finance (101); sales + retail (72); software_it + data_analytics (67); software_it + engineering (66); software_it + consulting (46); healthcare + science_research (45); marketing + digital_media (43); accounting + hr (37); healthcare + sales (37); construction + project_management (36); healthcare + teaching (34); software_it + project_management (33); healthcare + social_services_advocacy (29); data_analytics + science_research (28); sales + marketing (24)

## 4. Labels each resume category would receive (full deduplicated corpus, before sampling)

| Category | Resumes | Grade 2 | Grade 1 | Uncertain | Grade 0 (high conf.) | Grade 0 (low conf.) | Sample grade-2 titles |
|---|---|---|---|---|---|---|---|
| ACCOUNTANT | 118 | 634 | 792 | 131 | 9229 | 3284 | Senior Accountant; Billing Specialist; Billing Representative III (A/R) - Central Billing Office *Onsite - Las Vegas, NV*; Assistant Controller; Plant Controller; Controller |
| ADVOCATE | 118 | 116 | 278 | 0 | 10236 | 3440 | Lead Housing Coordinator; Industrial Base Outreach Analyst; Family Services Intake Clinician- Part Time; Case Worker, Crisis Intervention; Remote Licensed Clinical Social Worker - Ohio; Registered Nurse Case Manager |
| AGRICULTURE (excluded) | 63 | 10 | 284 | 0 | 10336 | 3440 | Senior Crop Specialist; Forester - Multiple Vacancies; Lead Landscape Designer/Architect; Grower Enrollment Specialist; Forestry Technician/ Inventory Forester; Landscape Designer |
| APPAREL | 97 | 108 | 503 | 0 | 10019 | 3440 | Apparel Sales Outfitter; Marketing Merchandising Manager; Full-time Nabisco Sales Service Representative/Merchandiser Order Writer; Beauty Advisor; Product Location Merchandiser; Seasonal Retail Sales Associate, Northridge Fashion Center |
| ARTS | 103 | 33 | 699 | 119 | 9870 | 3349 | Usher/Ticket Taker - Uptown Theater (Minneapolis); Public Art Manager; Sports MMJ-Photographer; Senior Illustrator; Corporate Sponsorship Specialist for Nationwide Art Battle Events & TV Show; Senior Gameplay Animator - XDefiant |
| AUTOMOBILE | 36 | 140 | 0 | 0 | 10490 | 3440 | Investment Banking Vice President \| Automotive M&A; Diesel Technician/Mechanic III - Entry Level; Diesel Mechanic Position Now Open!!!!!; Fleet & Equipment Specialist; NO 1 Auto Machine Mechanic; Automotive Installer |
| AVIATION | 117 | 49 | 615 | 0 | 9966 | 3440 | Mechanical Design Engineer (Aerospace Exp.); Aerospace VP of Sales and Marketing; Office Engineer II - Airport Operational Readiness - Training Coordinator - LAX Airport; Airport EOC Network Administrator (D); Airport Manager- Insider Threat Program; Operations Manager (STL Airport) |
| BANKING | 115 | 136 | 1272 | 119 | 9241 | 3302 | Relationship Banker - Woodmoor-Four Corners Financial Center - Rockville/Silver Spring Market - Silver Spring, MD; Senior Commercial Loan Officer; Relationship Banker- Carrboro; Commercial Loan Officer; Vice President of Compliance - The Tennessee Credit Union; Sr Loan Advisor |
| BPO | 22 | 195 | 366 | 184 | 9974 | 3351 | Customer Success Manager, Woopra; Customer Service Representative; Customer Service Representative; Customer Service Lead; Client Services Associate; Part-Time Patient Service Associate/Patient Access - Brooklyn |
| BUSINESS-DEVELOPMENT | 120 | 116 | 1926 | 773 | 8214 | 3041 | Business Development Manager; Business Development Manager - Outside Sales; Business Development Representative (French Speaking); Principal, AWS Strategy and Business Development; Business Development Manager; Business Development Manager |
| CHEF | 118 | 133 | 181 | 0 | 10316 | 3440 | Cook; Cook III; Line Cook; Food Service Worker; GUCCI Pastry Cook Osteria da Massimo Bottura - Beverly Hills; Restaurant Crewmember - Cook, Cashier, and Customer Service |
| CONSTRUCTION | 112 | 334 | 1104 | 0 | 9192 | 3440 | Supervisor Network Construction; Construction Cost Estimator; Shipyard Estimator and Contracts Administrator ; HVAC Truck Based Sr Controls Technician; Construction Manager; General Laborer |
| CONSULTANT | 115 | 75 | 782 | 114 | 9710 | 3389 | Technology Consulting Director; Productivity Consultant - Salaried Functions; SAP SD/MM OTC Consultant; Projects Assistant Consultant; Senior IAM Forgerock Consultant; Provider Partner Consultant, DentaQuest ( Baton Rouge, LA) |
| DESIGNER | 107 | 150 | 392 | 121 | 10058 | 3349 | UX Researcher, Pixel; UI/UX Designer; Junior Graphic Designer; User Experience Design Specialist; ASSOCIATE DESIGNER; Interaction Designer, Google Cloud |
| DIGITAL-MEDIA | 96 | 172 | 739 | 308 | 9625 | 3226 | Digital Marketing Measurement Analyst  ; Managing Editor; Social Media Manager; Strategic Communications Specialist and Technical Editor; Videographer; EXECUTIVE NEWS PRODUCER (NIGHTSIDE) - WANF |
| ENGINEERING | 118 | 506 | 2775 | 227 | 7623 | 2939 | Intern- DRAM Design Engineer; Senior Test Engineer; Project Engineer; Test Engineer II; Mechanical Engineer; IT Collaboration Systems Engineer |
| FINANCE | 118 | 397 | 1029 | 131 | 9229 | 3284 | Management Controller; Patient Financial Specialist; Controller; Controller; Financial Advisor; Investment Officer |
| FITNESS | 117 | 25 | 0 | 0 | 10605 | 3440 | Wellness Coach; Fitness Studio Manager \| Culver City, CA; Permanent Staff Athletic Trainer job in Brunswick, GA - Make $21/hour; Exercise Physiologist; Fitness Trainer; Fitness Trainer |
| HEALTHCARE | 115 | 1210 | 529 | 53 | 9018 | 3260 | Patient Account Registrar PD - Admitting; Hospice LPN Liaison ; Ambulatory Nursing Educator; Case Manager III, Group Therapy; Behavioral Health Counselor; MRI Manager |
| HR | 110 | 275 | 121 | 35 | 10245 | 3394 | Recruiting Strategist; HR Generalist; Compensation Consulting - Engagement Leader; Recruitment Sourcing Specialist; Seasonal Bilingual Talent Acquisition Associate; HR Coordinator - Therma |
| INFORMATION-TECHNOLOGY | 120 | 1382 | 1781 | 832 | 7435 | 2640 | Sr. Security Engineer , Robotics Security; Data Privacy analyst (NOT ATTORNEY) with 1-3 years of experience ONLY not a BI/technical role please no data architects/developers; Senior Software Engineer, Operations Microsoft .Net (ASP.NET, C#, JavaScript, XML); SAP FICO; Software Development Engineer II, IT Product and Development; Software Engineering Manager |
| PUBLIC-RELATIONS | 111 | 57 | 584 | 230 | 9903 | 3296 | Communications Manager; Strategic Communications Specialist, Junior; Public Affairs Specialist; Marketing Communications Manager; Sr. Communication Manager; Marketing Communications Specialist - 75484 |
| SALES | 116 | 967 | 1087 | 793 | 8182 | 3041 | Sales Development Representative - Remote - USA; Account Manager, East Coast; Outside Sales Representative; Financial Advisor - Sales Advisor - Investment Services; Director of Sales, Commerce; Retail Sales Leader Footwear |
| TEACHER | 102 | 156 | 105 | 52 | 10377 | 3380 | Assistant Professor, Questrom School of Business, Finance; Reading Tutor; Assistant Preschool Director; Classroom Teacher; High School Spanish Teacher; Director, Online Education Programs |

### Jobs per grading rule

| Category | title_primary_code_match | title_primary_family_uncoded | title_primary_no_job_codes | title_primary_code_mismatch | title_related | code_only | title_unrelated_code_match | title_unrelated | no_evidence |
|---|---|---|---|---|---|---|---|---|---|
| ACCOUNTANT | 610 | 0 | 24 | 85 | 551 | 156 | 131 | 9229 | 3284 |
| ADVOCATE | 0 | 116 | 0 | 0 | 278 | 0 | 0 | 10236 | 3440 |
| AGRICULTURE (excluded) | 0 | 10 | 0 | 0 | 284 | 0 | 0 | 10336 | 3440 |
| APPAREL | 0 | 108 | 0 | 0 | 503 | 0 | 0 | 10019 | 3440 |
| ARTS | 25 | 0 | 8 | 17 | 591 | 91 | 119 | 9870 | 3349 |
| AUTOMOBILE | 0 | 140 | 0 | 0 | 0 | 0 | 0 | 10490 | 3440 |
| AVIATION | 0 | 49 | 0 | 0 | 615 | 0 | 0 | 9966 | 3440 |
| BANKING | 114 | 0 | 22 | 50 | 1084 | 138 | 119 | 9241 | 3302 |
| BPO | 46 | 0 | 149 | 88 | 189 | 89 | 184 | 9974 | 3351 |
| BUSINESS-DEVELOPMENT | 106 | 0 | 10 | 19 | 1508 | 399 | 773 | 8214 | 3041 |
| CHEF | 0 | 133 | 0 | 0 | 181 | 0 | 0 | 10316 | 3440 |
| CONSTRUCTION | 0 | 334 | 0 | 0 | 1104 | 0 | 0 | 9192 | 3440 |
| CONSULTANT | 49 | 0 | 26 | 195 | 536 | 51 | 114 | 9710 | 3389 |
| DESIGNER | 136 | 0 | 14 | 58 | 243 | 91 | 121 | 10058 | 3349 |
| DIGITAL-MEDIA | 155 | 0 | 17 | 39 | 486 | 214 | 308 | 9625 | 3226 |
| ENGINEERING | 480 | 0 | 26 | 115 | 2159 | 501 | 227 | 7623 | 2939 |
| FINANCE | 368 | 0 | 29 | 91 | 782 | 156 | 131 | 9229 | 3284 |
| FITNESS | 0 | 25 | 0 | 0 | 0 | 0 | 0 | 10605 | 3440 |
| HEALTHCARE | 1032 | 0 | 178 | 264 | 85 | 180 | 53 | 9018 | 3260 |
| HR | 258 | 0 | 17 | 75 | 0 | 46 | 35 | 10245 | 3394 |
| INFORMATION-TECHNOLOGY | 1258 | 0 | 124 | 201 | 780 | 800 | 832 | 7435 | 2640 |
| PUBLIC-RELATIONS | 49 | 0 | 8 | 8 | 432 | 144 | 230 | 9903 | 3296 |
| SALES | 869 | 0 | 98 | 63 | 625 | 399 | 793 | 8182 | 3041 |
| TEACHER | 128 | 0 | 28 | 45 | 0 | 60 | 52 | 10377 | 3380 |

