import asyncio
from typing import Any, ClassVar
from urllib.parse import urlparse

import aiohttp

from .analyzer import TopicAnalyzer
from .error_handler import ErrorHandler


class GitHubAnalyzer:
    """Analyzer for GitHub repositories that processes files and generates topics."""

    CORE_FILES: ClassVar[list[str]] = ["README.md", "requirements.txt", "pyproject.toml", "package.json", "main.py", "app.py", "train.py"]

    def __init__(self):
        """Initialize the GitHubAnalyzer with base URL and required components."""
        self.base_url = "https://raw.githubusercontent.com"
        self.topic_analyzer = TopicAnalyzer()
        self.error_handler = ErrorHandler()

    def set_device(self, device: str):
        """Set the device for the topic analyzer.

        Args:
            device: Device to use ('cpu' or 'cuda')

        """
        self.topic_analyzer.set_device(device)

    def parse_github_url(self, url: str) -> tuple[str, str, str]:
        """Parse GitHub URL into components.

        Args:
            url: GitHub repository URL

        Returns:
            Tuple containing (owner, repo, branch)

        Raises:
            ValueError: If URL format is invalid

        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                return self.error_handler.handle_github_url_error(url, "URL must contain owner and repository")

            owner = path_parts[0]
            repo = path_parts[1]
            branch = "main"  # default branch

            return owner, repo, branch
        except Exception as e:
            return self.error_handler.handle_github_url_error(url, str(e))

    async def _fetch_file(self, session: aiohttp.ClientSession, url: str) -> str | None:
        """Fetch a single file content from GitHub.

        Args:
            session: aiohttp client session
            url: URL of the file to fetch

        Returns:
            File content or None if fetch fails

        """
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                return None
        except Exception:
            return None

    async def _fetch_core_files(self, repo_url: str) -> dict[str, str]:
        owner, repo, branch = self.parse_github_url(repo_url)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for file in self.CORE_FILES:
                url = f"{self.base_url}/{owner}/{repo}/{branch}/{file}"
                tasks.append(self._fetch_file(session, url))
            results = await asyncio.gather(*tasks)
            return {file: content for file, content in zip(self.CORE_FILES, results, strict=False) if content}

    def _parse_poetry_deps(self, content: str) -> list[str]:
        """Parse dependencies from pyproject.toml content.

        Args:
            content: Content of pyproject.toml file

        Returns:
            List of dependency names

        """
        deps = set()
        in_deps_section = False

        for line in content.split("\n"):
            line = line.strip()

            # Check if we're entering the dependencies section
            if "[tool.poetry.dependencies]" in line:
                in_deps_section = True
                continue

            # Check if we're exiting the dependencies section
            if in_deps_section and line.startswith("["):
                in_deps_section = False
                continue

            # Parse dependency line if we're in the dependencies section
            if in_deps_section and "=" in line:
                # Handle different poetry dependency formats
                package = line.split("=")[0].strip()
                # Remove quotes if present
                package = package.strip("\"'")

                # Skip python dependency
                if package.lower() != "python":
                    deps.add(package)

        return list(deps)

    async def _analyze_dependencies(self, files_content: dict[str, str]) -> list[str]:
        """Extract dependencies from requirement files.

        Args:
            files_content: Dictionary of file contents

        Returns:
            List of dependency names from all requirements files

        """
        deps = set()

        # Parse requirements.txt
        if "requirements.txt" in files_content:
            for line in files_content["requirements.txt"].split("\n"):
                if line and not line.startswith("#"):
                    package = line.split("==")[0].split(">=")[0].strip()
                    deps.add(package)

        # Parse pyproject.toml
        if "pyproject.toml" in files_content:
            content = files_content["pyproject.toml"]
            if "[tool.poetry.dependencies]" in content:
                deps.update(self._parse_poetry_deps(content))

        # Parse package.json
        if "package.json" in files_content:
            try:
                import json

                pkg_json = json.loads(files_content["package.json"])
                deps.update(pkg_json.get("dependencies", {}).keys())
                deps.update(pkg_json.get("devDependencies", {}).keys())
            except json.JSONDecodeError:
                pass

        return list(deps)

    async def analyze_repository(self, repo_url: str, category: str, subcategory: str) -> dict[str, Any]:
        """Analyze repository and generate comprehensive topics.

        Args:
            repo_url: GitHub repository URL
            category: Main category for topic classification
            subcategory: Sub-category for topic classification

        Returns:
            Dictionary containing analysis results including topics and dependencies

        """
        try:
            files_content = await self._fetch_core_files(repo_url)
            if not files_content:
                return self.error_handler.handle_file_fetch_error(repo_url, "No core files found")

            # Analyze README content
            readme_topics = []
            if "README.md" in files_content:
                readme_topics = await self.topic_analyzer.generate_topics(files_content["README.md"], category, subcategory)

            # Get dependencies
            dependencies = await self._analyze_dependencies(files_content)

            # Analyze Python files content
            code_content = ""
            for file in ["main.py", "app.py", "train.py"]:
                if file in files_content:
                    code_content += files_content[file] + "\n"

            code_topics = []
            if code_content:
                code_topics = await self.topic_analyzer.generate_topics(code_content, category, subcategory)

            return self.error_handler.success_response(
                {"readme_topics": readme_topics, "code_topics": code_topics, "dependencies": dependencies}
            )

        except Exception as e:
            return self.error_handler.handle_topic_analysis_error(
                str(e), {"repo_url": repo_url, "category": category, "subcategory": subcategory}
            )
