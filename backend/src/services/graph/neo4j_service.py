"""Neo4j Graph Database Service for Biomedical Knowledge Graph"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PaperNode:
    """Paper node data"""
    pmid: str
    title: str
    abstract: str
    journal: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None


@dataclass
class AuthorNode:
    """Author node data"""
    name: str
    affiliation: Optional[str] = None


@dataclass
class KeywordNode:
    """Keyword/MeSH term node data"""
    term: str
    term_type: str = "keyword"  # keyword, mesh_term


@dataclass
class GraphRelationship:
    """Relationship data"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None


class Neo4jService:
    """Service for Neo4j graph database operations"""

    def __init__(self):
        self._driver = None
        self._async_driver = None

    def _get_driver(self):
        """Get or create synchronous driver"""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                # Verify connectivity
                self._driver.verify_connectivity()
                logger.info("Neo4j connection established")
            except (ServiceUnavailable, AuthError) as e:
                logger.warning(f"Neo4j connection failed: {e}")
                self._driver = None
                raise
        return self._driver

    async def _get_async_driver(self):
        """Get or create async driver"""
        if self._async_driver is None:
            try:
                self._async_driver = AsyncGraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                logger.info("Neo4j async connection established")
            except (ServiceUnavailable, AuthError) as e:
                logger.warning(f"Neo4j async connection failed: {e}")
                self._async_driver = None
                raise
        return self._async_driver

    def close(self):
        """Close the driver connections"""
        if self._driver:
            self._driver.close()
            self._driver = None
        if self._async_driver:
            self._async_driver.close()
            self._async_driver = None

    def is_connected(self) -> bool:
        """Check if Neo4j is connected"""
        try:
            driver = self._get_driver()
            driver.verify_connectivity()
            return True
        except Exception:
            return False

    # ============== Schema Setup ==============

    def setup_schema(self):
        """Create indexes and constraints for optimal performance"""
        driver = self._get_driver()

        constraints_and_indexes = [
            # Constraints (unique)
            "CREATE CONSTRAINT paper_pmid IF NOT EXISTS FOR (p:Paper) REQUIRE p.pmid IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT keyword_term IF NOT EXISTS FOR (k:Keyword) REQUIRE k.term IS UNIQUE",
            "CREATE CONSTRAINT journal_name IF NOT EXISTS FOR (j:Journal) REQUIRE j.name IS UNIQUE",

            # Indexes for search performance
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX paper_date IF NOT EXISTS FOR (p:Paper) ON (p.publication_date)",
            "CREATE INDEX keyword_type IF NOT EXISTS FOR (k:Keyword) ON (k.term_type)",
        ]

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Schema query skipped (may already exist): {e}")

        logger.info("Neo4j schema setup completed")

    # ============== Paper Operations ==============

    def create_paper(self, paper: PaperNode) -> bool:
        """Create or update a paper node"""
        driver = self._get_driver()

        query = """
        MERGE (p:Paper {pmid: $pmid})
        SET p.title = $title,
            p.abstract = $abstract,
            p.journal = $journal,
            p.publication_date = $publication_date,
            p.doi = $doi,
            p.updated_at = datetime()
        RETURN p
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                query,
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                journal=paper.journal,
                publication_date=paper.publication_date,
                doi=paper.doi
            )
            return result.single() is not None

    def create_paper_with_relationships(
        self,
        paper: PaperNode,
        authors: List[str],
        keywords: List[str],
        mesh_terms: List[str] = None
    ) -> bool:
        """Create paper with all its relationships in a single transaction"""
        driver = self._get_driver()

        query = """
        // Create Paper node
        MERGE (p:Paper {pmid: $pmid})
        SET p.title = $title,
            p.abstract = $abstract,
            p.publication_date = $publication_date,
            p.doi = $doi,
            p.updated_at = datetime()

        // Create Journal and relationship
        WITH p
        FOREACH (j IN CASE WHEN $journal IS NOT NULL THEN [$journal] ELSE [] END |
            MERGE (journal:Journal {name: j})
            MERGE (p)-[:PUBLISHED_IN]->(journal)
        )

        // Create Authors and relationships
        WITH p
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (p)-[:AUTHORED_BY]->(a)

        // Create Keywords and relationships
        WITH p
        UNWIND $keywords AS keyword
        MERGE (k:Keyword {term: keyword})
        ON CREATE SET k.term_type = 'keyword'
        MERGE (p)-[:HAS_KEYWORD]->(k)

        // Create MeSH terms and relationships
        WITH p
        UNWIND $mesh_terms AS mesh
        MERGE (m:Keyword {term: mesh})
        ON CREATE SET m.term_type = 'mesh_term'
        MERGE (p)-[:HAS_MESH_TERM]->(m)

        RETURN p
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                query,
                pmid=paper.pmid,
                title=paper.title,
                abstract=paper.abstract,
                journal=paper.journal,
                publication_date=paper.publication_date,
                doi=paper.doi,
                authors=authors or [],
                keywords=keywords or [],
                mesh_terms=mesh_terms or []
            )
            return result.single() is not None

    def get_paper(self, pmid: str) -> Optional[Dict]:
        """Get paper with all its relationships"""
        driver = self._get_driver()

        query = """
        MATCH (p:Paper {pmid: $pmid})
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
        OPTIONAL MATCH (p)-[:HAS_MESH_TERM]->(m:Keyword)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(j:Journal)
        RETURN p,
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT k.term) as keywords,
               collect(DISTINCT m.term) as mesh_terms,
               j.name as journal
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, pmid=pmid)
            record = result.single()

            if not record:
                return None

            paper = dict(record["p"])
            paper["authors"] = record["authors"]
            paper["keywords"] = record["keywords"]
            paper["mesh_terms"] = record["mesh_terms"]
            paper["journal"] = record["journal"]

            return paper

    # ============== Relationship Operations ==============

    def create_citation(self, citing_pmid: str, cited_pmid: str) -> bool:
        """Create a citation relationship between papers"""
        driver = self._get_driver()

        query = """
        MATCH (citing:Paper {pmid: $citing_pmid})
        MATCH (cited:Paper {pmid: $cited_pmid})
        MERGE (citing)-[:CITES]->(cited)
        RETURN citing, cited
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, citing_pmid=citing_pmid, cited_pmid=cited_pmid)
            return result.single() is not None

    def create_similarity_relationship(
        self,
        pmid1: str,
        pmid2: str,
        similarity_score: float,
        similarity_type: str = "semantic"
    ) -> bool:
        """Create similarity relationship between papers"""
        driver = self._get_driver()

        query = """
        MATCH (p1:Paper {pmid: $pmid1})
        MATCH (p2:Paper {pmid: $pmid2})
        MERGE (p1)-[r:SIMILAR_TO]-(p2)
        SET r.score = $score,
            r.type = $sim_type,
            r.updated_at = datetime()
        RETURN p1, p2
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(
                query,
                pmid1=pmid1,
                pmid2=pmid2,
                score=similarity_score,
                sim_type=similarity_type
            )
            return result.single() is not None

    # ============== Graph Queries ==============

    def find_related_papers(
        self,
        pmid: str,
        limit: int = 10,
        relationship_types: List[str] = None
    ) -> List[Dict]:
        """Find papers related through shared authors, keywords, or citations"""
        driver = self._get_driver()

        if relationship_types is None:
            relationship_types = ["AUTHORED_BY", "HAS_KEYWORD", "CITES", "SIMILAR_TO"]

        query = """
        MATCH (p:Paper {pmid: $pmid})

        // Find papers by shared authors
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(related1:Paper)
        WHERE related1.pmid <> $pmid

        // Find papers by shared keywords
        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(related2:Paper)
        WHERE related2.pmid <> $pmid

        // Find papers by citation
        OPTIONAL MATCH (p)-[:CITES]-(related3:Paper)

        // Find papers by similarity
        OPTIONAL MATCH (p)-[sim:SIMILAR_TO]-(related4:Paper)

        WITH p,
             collect(DISTINCT {paper: related1, type: 'coauthor', score: 0.8}) +
             collect(DISTINCT {paper: related2, type: 'keyword', score: 0.6}) +
             collect(DISTINCT {paper: related3, type: 'citation', score: 0.9}) +
             collect(DISTINCT {paper: related4, type: 'semantic', score: sim.score}) as all_related

        UNWIND all_related as rel
        WHERE rel.paper IS NOT NULL

        WITH rel.paper as paper, rel.type as rel_type, MAX(rel.score) as score
        RETURN DISTINCT paper.pmid as pmid,
               paper.title as title,
               paper.abstract as abstract,
               rel_type as relationship_type,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, pmid=pmid, limit=limit)
            return [dict(record) for record in result]

    def find_coauthors(self, author_name: str, limit: int = 20) -> List[Dict]:
        """Find co-authors of a given author"""
        driver = self._get_driver()

        query = """
        MATCH (a:Author {name: $author_name})<-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]->(coauthor:Author)
        WHERE coauthor.name <> $author_name
        WITH coauthor, count(p) as collaborations, collect(p.title) as papers
        RETURN coauthor.name as name,
               collaborations,
               papers[0..5] as sample_papers
        ORDER BY collaborations DESC
        LIMIT $limit
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, author_name=author_name, limit=limit)
            return [dict(record) for record in result]

    def find_trending_keywords(self, limit: int = 20) -> List[Dict]:
        """Find most connected keywords (trending topics)"""
        driver = self._get_driver()

        query = """
        MATCH (k:Keyword)<-[:HAS_KEYWORD|HAS_MESH_TERM]-(p:Paper)
        WITH k, count(p) as paper_count
        RETURN k.term as keyword,
               k.term_type as type,
               paper_count
        ORDER BY paper_count DESC
        LIMIT $limit
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, limit=limit)
            return [dict(record) for record in result]

    def get_author_network(self, author_name: str, depth: int = 2) -> Dict:
        """Get author collaboration network"""
        driver = self._get_driver()

        query = """
        MATCH path = (a:Author {name: $author_name})<-[:AUTHORED_BY]-(:Paper)-[:AUTHORED_BY]->(coauthor:Author)
        WHERE coauthor.name <> $author_name
        WITH a, coauthor, count(*) as weight
        RETURN a.name as source,
               collect({target: coauthor.name, weight: weight}) as connections
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, author_name=author_name)
            record = result.single()

            if not record:
                return {"source": author_name, "connections": []}

            return {
                "source": record["source"],
                "connections": record["connections"]
            }

    def get_keyword_network(self, keyword: str, limit: int = 20) -> Dict:
        """Get keyword co-occurrence network"""
        driver = self._get_driver()

        query = """
        MATCH (k1:Keyword {term: $keyword})<-[:HAS_KEYWORD|HAS_MESH_TERM]-(p:Paper)-[:HAS_KEYWORD|HAS_MESH_TERM]->(k2:Keyword)
        WHERE k2.term <> $keyword
        WITH k2, count(p) as co_occurrence
        RETURN k2.term as keyword,
               k2.term_type as type,
               co_occurrence
        ORDER BY co_occurrence DESC
        LIMIT $limit
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, keyword=keyword, limit=limit)
            related = [dict(record) for record in result]

            return {
                "keyword": keyword,
                "related_keywords": related
            }

    def find_path_between_papers(self, pmid1: str, pmid2: str, max_depth: int = 4) -> List[Dict]:
        """Find shortest path between two papers"""
        driver = self._get_driver()

        query = """
        MATCH path = shortestPath(
            (p1:Paper {pmid: $pmid1})-[*1..$max_depth]-(p2:Paper {pmid: $pmid2})
        )
        RETURN [node in nodes(path) |
            CASE
                WHEN 'Paper' IN labels(node) THEN {type: 'Paper', pmid: node.pmid, title: node.title}
                WHEN 'Author' IN labels(node) THEN {type: 'Author', name: node.name}
                WHEN 'Keyword' IN labels(node) THEN {type: 'Keyword', term: node.term}
                WHEN 'Journal' IN labels(node) THEN {type: 'Journal', name: node.name}
                ELSE {type: 'Unknown'}
            END
        ] as path_nodes,
        [rel in relationships(path) | type(rel)] as relationship_types,
        length(path) as path_length
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query, pmid1=pmid1, pmid2=pmid2, max_depth=max_depth)
            record = result.single()

            if not record:
                return []

            return {
                "path_nodes": record["path_nodes"],
                "relationship_types": record["relationship_types"],
                "path_length": record["path_length"]
            }

    # ============== Statistics ==============

    def get_stats(self) -> Dict:
        """Get graph database statistics"""
        driver = self._get_driver()

        query = """
        MATCH (p:Paper) WITH count(p) as papers
        MATCH (a:Author) WITH papers, count(a) as authors
        MATCH (k:Keyword) WITH papers, authors, count(k) as keywords
        MATCH (j:Journal) WITH papers, authors, keywords, count(j) as journals
        MATCH ()-[r]->() WITH papers, authors, keywords, journals, count(r) as relationships
        RETURN papers, authors, keywords, journals, relationships
        """

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run(query)
            record = result.single()

            if not record:
                return {
                    "papers": 0,
                    "authors": 0,
                    "keywords": 0,
                    "journals": 0,
                    "relationships": 0,
                    "status": "empty"
                }

            return {
                "papers": record["papers"],
                "authors": record["authors"],
                "keywords": record["keywords"],
                "journals": record["journals"],
                "relationships": record["relationships"],
                "status": "connected"
            }

    def clear_database(self) -> bool:
        """Clear all nodes and relationships (use with caution!)"""
        driver = self._get_driver()

        query = "MATCH (n) DETACH DELETE n"

        with driver.session(database=settings.NEO4J_DATABASE) as session:
            session.run(query)
            return True


# Singleton instance
_neo4j_service: Optional[Neo4jService] = None


def get_neo4j_service() -> Neo4jService:
    """Get or create Neo4j service instance"""
    global _neo4j_service
    if _neo4j_service is None:
        _neo4j_service = Neo4jService()
    return _neo4j_service
