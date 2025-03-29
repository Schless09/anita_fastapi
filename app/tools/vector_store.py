from app.config.services import get_embeddings
from app.config.logger import logger

class VectorStoreTool:
    def __init__(self, jobs_index: Index, candidates_index: Index):
        logger.info("Initializing VectorStoreTool with provided indices")
        self.jobs_index = jobs_index
        self.candidates_index = candidates_index
        logger.info("Attempting to initialize get_embeddings function")
        try:
            self.get_embeddings = get_embeddings
            logger.info("✅ get_embeddings function initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize get_embeddings: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise
        logger.info("VectorStoreTool initialization complete")

    async def upsert_job(self, job_id: str, job_description: str):
        logger.info(f"Upserting job with ID: {job_id}")
        try:
            embeddings = await self.get_embeddings(job_description)
            logger.info(f"✅ Generated embeddings for job {job_id}")
            
            self.jobs_index.upsert(vectors=[(job_id, embeddings, {"text": job_description})])
            logger.info(f"✅ Successfully upserted job {job_id} to vector store")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error upserting job {job_id}: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise

    async def upsert_candidate(self, candidate_id: str, candidate_description: str):
        logger.info(f"Upserting candidate with ID: {candidate_id}")
        try:
            embeddings = await self.get_embeddings(candidate_description)
            logger.info(f"✅ Generated embeddings for candidate {candidate_id}")
            
            self.candidates_index.upsert(vectors=[(candidate_id, embeddings, {"text": candidate_description})])
            logger.info(f"✅ Successfully upserted candidate {candidate_id} to vector store")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error upserting candidate {candidate_id}: {str(e)}")
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            raise 