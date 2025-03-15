"""
Модуль для детального анализа исходного кода

Предоставляет функциональность для:
1. Анализа структуры кода (классы, функции, методы)
2. Построения графа зависимостей между файлами
3. Извлечения метаданных из файлов различных типов
"""

import ast
import re
import os
import importlib.util
from typing import Dict, List, Set, Optional, Tuple, Any, Union


class CodeAnalyzer:
    """
    Анализатор структуры исходного кода для различных языков программирования
    """

    def __init__(self, file_path: str, content: str = None):
        """
        Инициализирует анализатор кода

        Args:
            file_path: путь к файлу
            content: содержимое файла (опционально)
        """
        self.file_path = file_path
        self.content = content
        self._parsed_ast = None
        self._classes = None
        self._functions = None
        self._imports = None

    def parse(self) -> Optional[Dict[str, Any]]:
        """
        Парсит код файла в зависимости от его типа

        Returns:
            Optional[Dict[str, Any]]: структура кода или None в случае ошибки
        """
        if not self.content:
            try:
                with open (self.file_path, 'r', encoding='utf-8') as f:
                    self.content = f.read ()
            except Exception as e:
                return {"error": f"Не удалось прочитать файл: {str (e)}"}

        # Определяем тип файла по расширению
        if self.file_path.endswith ('.py'):
            return self._parse_python ()
        elif self.file_path.endswith (('.js', '.jsx', '.ts', '.tsx')):
            return self._parse_javascript ()
        elif self.file_path.endswith (('.java')):
            return self._parse_java ()
        elif self.file_path.endswith (('.c', '.cpp', '.h', '.hpp')):
            return self._parse_cpp ()
        elif self.file_path.endswith (('.sol')):
            return self._parse_solidity ()
        elif self.file_path.endswith (('.go')):
            return self._parse_golang ()
        elif self.file_path.endswith (('.rs')):
            return self._parse_rust ()
        else:
            # Попытка определить тип файла на основе его содержимого
            return self._parse_generic ()

    def _parse_python(self) -> Dict[str, Any]:
        """
        Анализирует Python файл с использованием AST

        Returns:
            Dict[str, Any]: структура кода
        """
        try:
            self._parsed_ast = ast.parse (self.content)
            return self._extract_python_structure ()
        except SyntaxError as e:
            return {"error": f"Невозможно распарсить Python код - ошибка синтаксиса: {str (e)}"}
        except Exception as e:
            return {"error": f"Ошибка при анализе Python кода: {str (e)}"}

    def _extract_python_structure(self) -> Dict[str, Any]:
        """
        Извлекает структуру из Python AST

        Returns:
            Dict[str, Any]: структура кода
        """
        visitor = PythonCodeVisitor ()
        visitor.visit (self._parsed_ast)

        return {
            "language": "python",
            "imports": visitor.imports,
            "classes": visitor.classes,
            "functions": visitor.functions,
            "variables": visitor.variables,
            "dependencies": list (visitor.dependencies)
        }

    def _parse_javascript(self) -> Dict[str, Any]:
        """
        Анализирует JavaScript/TypeScript файл с помощью регулярных выражений

        Returns:
            Dict[str, Any]: структура кода
        """
        imports = []
        classes = []
        functions = []
        dependencies = set ()

        # Поиск импортов
        import_regex = r'import\s+(?:{([^}]+)}|([^;]+))\s+from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer (import_regex, self.content):
            source = match.group (3)
            imports.append ({
                "source": source,
                "imported": match.group (1) or match.group (2)
            })
            dependencies.add (source)

        # Поиск классов
        class_regex = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer (class_regex, self.content):
            class_name = match.group (1)
            parent_class = match.group (2)

            # Поиск методов для класса
            methods = []
            method_regex = r'(?:async\s+)?(?:static\s+)?(?:get|set)?\s*(\w+)\s*\([^)]*\)\s*{(?:[^{}]|{[^{}]*})*}'
            class_start = match.end ()
            class_content = self._find_class_content (class_start)

            if class_content:
                for method_match in re.finditer (method_regex, class_content):
                    method_name = method_match.group (1)
                    if method_name not in ['constructor', 'if', 'for', 'while', 'switch']:
                        methods.append ({"name": method_name})

            classes.append ({
                "name": class_name,
                "extends": parent_class,
                "methods": methods
            })

        # Поиск функций
        function_regex = r'(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:\([^)]*\)|\([^)]*\)\s*=>\s*)'
        for match in re.finditer (function_regex, self.content):
            function_name = match.group (1)
            functions.append ({
                "name": function_name
            })

        return {
            "language": "javascript",
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "dependencies": list (dependencies)
        }

    def _find_class_content(self, start_pos: int) -> Optional[str]:
        """
        Находит содержимое класса на основе позиции открывающей скобки

        Args:
            start_pos: позиция в тексте после объявления класса

        Returns:
            Optional[str]: содержимое класса или None
        """
        open_brace_pattern = r'{'
        class_content = self.content[start_pos:]
        brace_match = re.search (open_brace_pattern, class_content)

        if brace_match:
            brace_pos = brace_match.start () + start_pos
            # Находим закрывающую скобку с учетом вложенности
            brace_level = 1
            for i in range (brace_pos + 1, len (self.content)):
                if self.content[i] == '{':
                    brace_level += 1
                elif self.content[i] == '}':
                    brace_level -= 1
                    if brace_level == 0:  # Найдена закрывающая скобка для impl
                        return self.content[brace_pos + 1:i]

        return None

    def _parse_java(self) -> Dict[str, Any]:
        """
        Анализирует Java файл с помощью регулярных выражений

        Returns:
            Dict[str, Any]: структура кода
        """
        imports = []
        classes = []
        interfaces = []
        dependencies = set ()
        package = None

        # Поиск пакета
        package_regex = r'package\s+([^;]+);'
        package_match = re.search (package_regex, self.content)
        if package_match:
            package = package_match.group (1)

        # Поиск импортов
        import_regex = r'import\s+([^;]+);'
        for match in re.finditer (import_regex, self.content):
            import_path = match.group (1)
            imports.append (import_path)
            dependencies.add (import_path)

        # Поиск классов
        class_regex = r'(?:public|private|protected)?\s+class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?'
        for match in re.finditer (class_regex, self.content):
            class_name = match.group (1)
            parent_class = match.group (2)

            implements = []
            if match.group (3):
                implements = [i.strip () for i in match.group (3).split (',')]

            classes.append ({
                "name": class_name,
                "extends": parent_class,
                "implements": implements
            })

        # Поиск интерфейсов
        interface_regex = r'(?:public|private|protected)?\s+interface\s+(\w+)(?:\s+extends\s+([^{]+))?'
        for match in re.finditer (interface_regex, self.content):
            interface_name = match.group (1)
            extends = []
            if match.group (2):
                extends = [i.strip () for i in match.group (2).split (',')]

            interfaces.append ({
                "name": interface_name,
                "extends": extends
            })

        return {
            "language": "java",
            "package": package,
            "imports": imports,
            "classes": classes,
            "interfaces": interfaces,
            "dependencies": list (dependencies)
        }

    def _parse_cpp(self) -> Dict[str, Any]:
        """
        Анализирует C/C++ файл с помощью регулярных выражений

        Returns:
            Dict[str, Any]: структура кода
        """
        includes = []
        classes = []
        functions = []
        dependencies = set ()

        # Поиск включений
        include_regex = r'#include\s+[<"]([^>"]+)[>"]'
        for match in re.finditer (include_regex, self.content):
            include_path = match.group (1)
            includes.append (include_path)
            dependencies.add (include_path)

        # Поиск классов
        class_regex = r'(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)?\s*(\w+))?'
        for match in re.finditer (class_regex, self.content):
            class_name = match.group (1)
            parent_class = match.group (2)

            classes.append ({
                "name": class_name,
                "parent": parent_class
            })

        # Поиск функций
        function_regex = r'(?:virtual\s+)?(?:static\s+)?(?:\w+(?:::\w+)?(?:<[^>]*>)?)\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer (function_regex, self.content):
            function_name = match.group (1)
            if function_name not in ['if', 'for', 'while', 'switch']:
                functions.append ({
                    "name": function_name
                })

        return {
            "language": "c++",
            "includes": includes,
            "classes": classes,
            "functions": functions,
            "dependencies": list (dependencies)
        }

    def _parse_solidity(self) -> Dict[str, Any]:
        """
        Анализирует Solidity файл (смарт-контракты) с помощью регулярных выражений

        Returns:
            Dict[str, Any]: структура кода
        """
        imports = []
        contracts = []
        interfaces = []
        dependencies = set ()
        pragma = None

        # Поиск pragma
        pragma_regex = r'pragma\s+solidity\s+([^;]+);'
        pragma_match = re.search (pragma_regex, self.content)
        if pragma_match:
            pragma = pragma_match.group (1)

        # Поиск импортов
        import_regex = r'import\s+(?:"([^"]+)"|\'([^\']+)\')'
        for match in re.finditer (import_regex, self.content):
            import_path = match.group (1) or match.group (2)
            imports.append (import_path)
            dependencies.add (import_path)

        # Поиск контрактов
        contract_regex = r'contract\s+(\w+)(?:\s+is\s+([^{]+))?'
        for match in re.finditer (contract_regex, self.content):
            contract_name = match.group (1)
            inherits = []
            if match.group (2):
                inherits = [i.strip () for i in match.group (2).split (',')]

            # Поиск функций в контракте
            start_pos = match.end ()
            contract_content = self._find_class_content (start_pos)
            functions = []

            if contract_content:
                function_regex = r'function\s+(\w+)\s*\([^)]*\)\s*(?:public|private|internal|external)?(?:\s+view|\s+pure|\s+payable)?\s*(?:returns\s*\([^)]*\))?\s*[{;]'
                for func_match in re.finditer (function_regex, contract_content):
                    functions.append ({"name": func_match.group (1)})

            contracts.append ({
                "name": contract_name,
                "inherits": inherits,
                "functions": functions
            })

        # Поиск интерфейсов
        interface_regex = r'interface\s+(\w+)(?:\s+is\s+([^{]+))?'
        for match in re.finditer (interface_regex, self.content):
            interface_name = match.group (1)
            inherits = []
            if match.group (2):
                inherits = [i.strip () for i in match.group (2).split (',')]

            interfaces.append ({
                "name": interface_name,
                "inherits": inherits
            })

        return {
            "language": "solidity",
            "pragma": pragma,
            "imports": imports,
            "contracts": contracts,
            "interfaces": interfaces,
            "dependencies": list (dependencies)
        }

    def _parse_golang(self) -> Dict[str, Any]:
        """
        Анализирует Go файл с помощью регулярных выражений

        Returns:
            Dict[str, Any]: структура кода
        """
        imports = []
        structs = []
        interfaces = []
        functions = []
        dependencies = set ()
        package = None

        # Поиск пакета
        package_regex = r'package\s+(\w+)'
        package_match = re.search (package_regex, self.content)
        if package_match:
            package = package_match.group (1)

        # Поиск импортов
        import_block_regex = r'import\s+\(\s*(.*?)\s*\)'
        import_block_match = re.search (import_block_regex, self.content, re.DOTALL)

        if import_block_match:
            # Блочный импорт
            import_content = import_block_match.group (1)
            for imp in re.finditer (r'(?:[\w._/]+\s+)?"([^"]+)"', import_content):
                import_path = imp.group (1)
                imports.append (import_path)
                dependencies.add (import_path)
        else:
            # Однострочный импорт
            single_import_regex = r'import\s+(?:[\w._/]+\s+)?"([^"]+)"'
            for match in re.finditer (single_import_regex, self.content):
                import_path = match.group (1)
                imports.append (import_path)
                dependencies.add (import_path)

        # Поиск структур
        struct_regex = r'type\s+(\w+)\s+struct\s*{'
        for match in re.finditer (struct_regex, self.content):
            struct_name = match.group (1)
            structs.append ({
                "name": struct_name
            })

        # Поиск интерфейсов
        interface_regex = r'type\s+(\w+)\s+interface\s*{'
        for match in re.finditer (interface_regex, self.content):
            interface_name = match.group (1)
            interfaces.append ({
                "name": interface_name
            })

        # Поиск функций
        function_regex = r'func\s+(?:\(\s*\w+\s+\*?\w+\s*\)\s+)?(\w+)\s*\([^)]*\)'
        for match in re.finditer (function_regex, self.content):
            function_name = match.group (1)
            functions.append ({
                "name": function_name
            })

        return {
            "language": "go",
            "package": package,
            "imports": imports,
            "structs": structs,
            "interfaces": interfaces,
            "functions": functions,
            "dependencies": list (dependencies)
        }

    def _parse_rust(self) -> Dict[str, Any]:
        """
        Анализирует Rust файл с помощью регулярных выражений

        Returns:
            Dict[str, Any]: структура кода
        """
        imports = []
        structs = []
        enums = []
        impls = []
        traits = []
        functions = []
        dependencies = set ()

        # Поиск импортов
        use_regex = r'use\s+([^;]+);'
        for match in re.finditer (use_regex, self.content):
            use_path = match.group (1)
            imports.append (use_path)
            # Извлекаем зависимость (первую часть пути)
            path_parts = use_path.split ('::')[0]
            dependencies.add (path_parts)

        # Поиск структур
        struct_regex = r'struct\s+(\w+)(?:<[^>]*>)?\s*(?:\([^)]*\)|{)'
        for match in re.finditer (struct_regex, self.content):
            struct_name = match.group (1)
            structs.append ({
                "name": struct_name
            })

        # Поиск перечислений
        enum_regex = r'enum\s+(\w+)(?:<[^>]*>)?\s*{'
        for match in re.finditer (enum_regex, self.content):
            enum_name = match.group (1)
            enums.append ({
                "name": enum_name
            })

        # Поиск реализаций
        impl_regex = r'impl(?:<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)(?:<[^>]*>)?\s*{'
        for match in re.finditer (impl_regex, self.content):
            trait_name = match.group (1)
            type_name = match.group (2)

            # Поиск методов
            start_pos = match.end ()
            impl_content = self._find_class_content (start_pos)
            methods = []

            if impl_content:
                method_regex = r'(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)'
                for method_match in re.finditer (method_regex, impl_content):
                    methods.append ({"name": method_match.group (1)})

            impls.append ({
                "type": type_name,
                "trait": trait_name,
                "methods": methods
            })

        # Поиск трейтов
        trait_regex = r'trait\s+(\w+)(?:<[^>]*>)?\s*(?::\s*[^{]+)?{'
        for match in re.finditer (trait_regex, self.content):
            trait_name = match.group (1)
            traits.append ({
                "name": trait_name
            })

        # Поиск функций
        function_regex = r'(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)'
        for match in re.finditer (function_regex, self.content):
            if not any (match.start () > start and match.end () < end for start, end in self._get_impl_ranges ()):
                function_name = match.group (1)
                functions.append ({
                    "name": function_name
                })

        return {
            "language": "rust",
            "imports": imports,
            "structs": structs,
            "enums": enums,
            "impls": impls,
            "traits": traits,
            "functions": functions,
            "dependencies": list (dependencies)
        }

    def _get_impl_ranges(self) -> List[Tuple[int, int]]:
        """
        Находит диапазоны реализаций в файле Rust

        Returns:
            List[Tuple[int, int]]: список пар (начало, конец) для каждой реализации
        """
        ranges = []
        impl_regex = r'impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?\w+(?:<[^>]*>)?\s*{'

        for match in re.finditer (impl_regex, self.content):
            start_pos = match.start ()
            # Находим закрывающую скобку
            brace_level = 0
            end_pos = None

            for i, char in enumerate (self.content[match.end ():], match.end ()):
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == -1:  # Найдена закрывающая скобка для impl
                        end_pos = i
                        break

            if end_pos:
                ranges.append ((start_pos, end_pos))

        return ranges

    def _parse_generic(self) -> Dict[str, Any]:
        """
        Анализирует файл неизвестного типа на основе его содержимого

        Returns:
            Dict[str, Any]: базовая структура кода
        """
        # Определяем возможный язык на основе содержимого
        language = self._detect_language ()

        # Базовый анализ
        functions = []
        classes = []

        # Обобщенные регулярные выражения для поиска функций
        function_patterns = [
            r'function\s+(\w+)\s*\(',  # JavaScript, PHP
            r'def\s+(\w+)\s*\(',  # Python
            r'sub\s+(\w+)\s*\{',  # Perl
            r'func\s+(\w+)\s*\(',  # Go
            r'void\s+(\w+)\s*\(',  # C, C++, Java
            r'int\s+(\w+)\s*\(',  # C, C++, Java
            r'string\s+(\w+)\s*\(',  # C, C++, Java
            r'static\s+\w+\s+(\w+)\s*\('  # C, C++, Java
        ]

        # Обобщенные регулярные выражения для поиска классов
        class_patterns = [
            r'class\s+(\w+)',  # Python, Java, C++, JavaScript
            r'interface\s+(\w+)',  # Java, TypeScript
            r'struct\s+(\w+)',  # C, C++, Go, Rust
            r'type\s+(\w+)\s+struct',  # Go
            r'trait\s+(\w+)',  # Rust
            r'module\s+(\w+)'  # Ruby
        ]

        # Поиск функций
        for pattern in function_patterns:
            for match in re.finditer (pattern, self.content):
                function_name = match.group (1)
                if function_name not in [f["name"] for f in functions]:
                    functions.append ({"name": function_name})

        # Поиск классов
        for pattern in class_patterns:
            for match in re.finditer (pattern, self.content):
                class_name = match.group (1)
                if class_name not in [c["name"] for c in classes]:
                    classes.append ({"name": class_name})

        return {
            "language": language,
            "functions": functions,
            "classes": classes,
            "is_generic_analysis": True
        }

    def _detect_language(self) -> str:
        """
        Определяет язык программирования на основе содержимого файла

        Returns:
            str: предполагаемый язык программирования
        """
        # Простая эвристика для определения языка
        if re.search (r'^\s*<[?!]', self.content) or re.search (r'<html', self.content, re.IGNORECASE):
            return "html"
        elif re.search (r'import\s+React|<\/?[A-Z][A-Za-z]*>', self.content):
            return "jsx/tsx"
        elif re.search (r'import\s+[{[]\s*.*?\s*[}\]]\s+from', self.content):
            return "javascript/typescript"
        elif re.search (r'package\s+[\w.]+;|import\s+[\w.]+;|public\s+class', self.content):
            return "java"
        elif re.search (r'#include\s+<[\w.]+>|void\s+\w+\s*\(', self.content):
            return "c/c++"
        elif re.search (r'def\s+\w+\s*\(.*\):|import\s+\w+|from\s+\w+\s+import', self.content):
            return "python"
        elif re.search (r'package\s+\w+|func\s+\w+\s*\(|import\s+\(', self.content):
            return "go"
        elif re.search (r'use\s+\w+::|fn\s+\w+\s*\(|impl\s+', self.content):
            return "rust"
        elif re.search (r'pragma\s+solidity|contract\s+\w+|function\s+\w+\s*\(.*\)\s*public', self.content):
            return "solidity"
        else:
            return "unknown"

    def get_dependency_graph(self) -> Dict[str, Any]:
        """
        Возвращает граф зависимостей файла

        Returns:
            Dict[str, Any]: граф зависимостей или ошибку
        """
        result = self.parse ()
        if not result or "error" in result:
            return {"error": "Невозможно построить граф зависимостей"}

        dependencies = {}

        if "language" in result:
            language = result["language"]

            if language == "python":
                for imp in result.get ("imports", []):
                    name = imp.get ("name", "")
                    if name.startswith ("."):
                        # Относительный импорт
                        dependencies[name] = True
                    else:
                        # Внешний импорт
                        dependencies[name] = False

            elif language in ["javascript", "jsx/tsx"]:
                for imp in result.get ("imports", []):
                    source = imp.get ("source", "")
                    if source.startswith ("."):
                        dependencies[source] = True
                    else:
                        dependencies[source] = False

            elif language == "java":
                for imp in result.get ("imports", []):
                    if "." in imp:
                        package = imp.split (".")[0]
                        dependencies[package] = imp.startswith (".")
                    else:
                        dependencies[imp] = False

            elif language == "c/c++":
                for inc in result.get ("includes", []):
                    is_local = not (inc.startswith ("<") and inc.endswith (">"))
                    dependencies[inc] = is_local

            elif language == "go":
                for imp in result.get ("imports", []):
                    is_local = "/" in imp and not imp.startswith ("github.com/")
                    dependencies[imp] = is_local

            elif language == "rust":
                for imp in result.get ("imports", []):
                    parts = imp.split ("::")
                    if parts:
                        is_local = not parts[0] in ["std", "core", "alloc"]
                        dependencies[parts[0]] = is_local

            elif language == "solidity":
                for imp in result.get ("imports", []):
                    is_local = imp.startswith ("./") or imp.startswith ("../")
                    dependencies[imp] = is_local

        return dependencies


class PythonCodeVisitor (ast.NodeVisitor):
    """
    Посетитель узлов AST для извлечения информации о Python-коде
    """

    def __init__(self):
        self.imports = []
        self.classes = []
        self.functions = []
        self.variables = []
        self.dependencies = set ()
        self.current_class = None

    def visit_Import(self, node):
        for name in node.names:
            self.imports.append ({
                "name": name.name,
                "alias": name.asname
            })
            self.dependencies.add (name.name)
        self.generic_visit (node)

    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module
            self.dependencies.add (module)
            for name in node.names:
                self.imports.append ({
                    "module": module,
                    "name": name.name,
                    "alias": name.asname
                })
        self.generic_visit (node)

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name

        class_info = {
            "name": node.name,
            "methods": [],
            "bases": [self._get_name (base) for base in node.bases],
            "decorators": [self._get_name (d) for d in node.decorator_list]
        }

        # Сохраняем текущий класс для обработки его методов
        for item in node.body:
            if isinstance (item, ast.FunctionDef):
                method_info = self._process_function (item)
                method_info["is_method"] = True
                class_info["methods"].append (method_info)

        self.classes.append (class_info)
        self.current_class = old_class
        self.generic_visit (node)

    def visit_FunctionDef(self, node):
        if not self.current_class:  # Только функции верхнего уровня
            function_info = self._process_function (node)
            self.functions.append (function_info)
        self.generic_visit (node)

    def visit_AsyncFunctionDef(self, node):
        if not self.current_class:  # Только функции верхнего уровня
            function_info = self._process_function (node)
            function_info["is_async"] = True
            self.functions.append (function_info)
        self.generic_visit (node)

    def visit_Assign(self, node):
        # Обрабатываем только присваивания верхнего уровня
        if not self.current_class:
            for target in node.targets:
                if isinstance (target, ast.Name):
                    self.variables.append ({
                        "name": target.id,
                        "value_type": self._get_value_type (node.value)
                    })
        self.generic_visit (node)

    def _process_function(self, node):
        return {
            "name": node.name,
            "args": self._get_args (node.args),
            "decorators": [self._get_name (d) for d in node.decorator_list],
            "is_async": isinstance (node, ast.AsyncFunctionDef)
        }

    def _get_args(self, args):
        result = []
        # Аргументы
        for arg in args.args:
            arg_info = {"name": arg.arg}
            if hasattr (arg, 'annotation') and arg.annotation:
                arg_info["type"] = self._get_name (arg.annotation)
            result.append (arg_info)

        # Аргументы со значениями по умолчанию
        if args.defaults:
            defaults_offset = len (args.args) - len (args.defaults)
            for i, default in enumerate (args.defaults):
                result[defaults_offset + i]["default"] = self._get_default_value (default)

        return result

    def _get_name(self, node):
        if isinstance (node, ast.Name):
            return node.id
        elif isinstance (node, ast.Attribute):
            return f"{self._get_name (node.value)}.{node.attr}"
        elif isinstance (node, ast.Subscript):
            return f"{self._get_name (node.value)}[{self._get_name (node.slice)}]"
        elif isinstance (node, ast.Call):
            return f"{self._get_name (node.func)}()"
        elif isinstance (node, ast.Constant):
            return str (node.value)
        return str (node.__class__.__name__)

    def _get_default_value(self, node):
        if isinstance (node, ast.Constant):
            return node.value
        elif isinstance (node, (ast.List, ast.Tuple, ast.Set)):
            return [self._get_default_value (elt) for elt in node.elts]
        elif isinstance (node, ast.Dict):
            return {self._get_default_value (k): self._get_default_value (v)
                    for k, v in zip (node.keys, node.values)}
        return self._get_name (node)

    def _get_value_type(self, node):
        if isinstance (node, ast.Constant):
            return type (node.value).__name__
        elif isinstance (node, ast.List):
            return "list"
        elif isinstance (node, ast.Tuple):
            return "tuple"
        elif isinstance (node, ast.Dict):
            return "dict"
        elif isinstance (node, ast.Set):
            return "set"
        elif isinstance (node, ast.Name):
            return node.id
        elif isinstance (node, ast.Call):
            return self._get_name (node.func)
        return "unknown"


class ProjectAnalyzer:
    """
    Анализирует весь проект для построения зависимостей между файлами
    и сбора метаданных о проекте
    """

    def __init__(self, project_path: str, file_paths: List[str]):
        """
        Инициализирует анализатор проекта

        Args:
            project_path: корневой путь проекта
            file_paths: список относительных путей к файлам
        """
        self.project_path = project_path
        self.file_paths = file_paths
        self.file_analyzers = {}
        self.dependency_graph = {}

    def analyze_project(self) -> Dict[str, Any]:
        """
        Анализирует все файлы проекта

        Returns:
            Dict[str, Any]: сводка по проекту
        """
        print ("Анализ структуры файлов проекта...")

        # Анализируем каждый файл
        for file_path in self.file_paths:
            full_path = os.path.join (self.project_path, file_path)
            analyzer = CodeAnalyzer (full_path)
            result = analyzer.parse ()
            if result and "error" not in result:
                self.file_analyzers[file_path] = analyzer

        # Строим граф зависимостей
        self._build_dependency_graph ()

        # Генерируем сводку
        return self._generate_project_summary ()

    def _build_dependency_graph(self) -> None:
        """
        Строит граф зависимостей между файлами проекта
        """
        for file_path, analyzer in self.file_analyzers.items ():
            file_deps = analyzer.get_dependency_graph ()
            if "error" not in file_deps:
                self.dependency_graph[file_path] = file_deps

    def _generate_project_summary(self) -> Dict[str, Any]:
        """
        Генерирует сводную информацию о проекте

        Returns:
            Dict[str, Any]: сводка по проекту
        """
        # Подсчет типов файлов
        file_types = self._count_file_types ()

        # Сбор метаданных о языках программирования
        language_stats = self._collect_language_stats ()

        # Сбор зависимостей
        external_dependencies = self._extract_external_dependencies ()
        internal_dependencies = self._build_internal_dependency_graph ()

        summary = {
            "files": len (self.file_paths),
            "analyzed_files": len (self.file_analyzers),
            "file_types": file_types,
            "language_stats": language_stats,
            "class_count": self._count_classes (),
            "function_count": self._count_functions (),
            "external_dependencies": external_dependencies,
            "internal_dependencies": internal_dependencies
        }

        print (f"Анализ структуры проекта завершен. Проанализировано {summary['analyzed_files']} из {summary['files']} файлов.")
        return summary

    def _count_file_types(self) -> Dict[str, int]:
        """
        Подсчитывает типы файлов в проекте

        Returns:
            Dict[str, int]: количество файлов каждого типа
        """
        file_types = {}
        for path in self.file_paths:
            ext = path.split ('.')[-1] if '.' in path else 'unknown'
            file_types[ext] = file_types.get (ext, 0) + 1
        return file_types

    def _collect_language_stats(self) -> Dict[str, int]:
        """
        Собирает статистику по языкам программирования

        Returns:
            Dict[str, int]: количество файлов каждого языка
        """
        language_stats = {}
        for file_path, analyzer in self.file_analyzers.items ():
            result = analyzer.parse ()
            if result and "error" not in result and "language" in result:
                language = result["language"]
                language_stats[language] = language_stats.get (language, 0) + 1
        return language_stats

    def _count_classes(self) -> int:
        """
        Подсчитывает классы в проекте

        Returns:
            int: общее количество классов
        """
        count = 0
        for file_path, analyzer in self.file_analyzers.items ():
            result = analyzer.parse ()
            if result and "error" not in result:
                # Python классы
                count += len (result.get ("classes", []))
                # Java, JavaScript классы
                count += len (result.get ("interfaces", []))
                # Solidity контракты
                count += len (result.get ("contracts", []))
                # Go структуры
                count += len (result.get ("structs", []))
                # Rust трейты
                count += len (result.get ("traits", []))
                # Rust перечисления
                count += len (result.get ("enums", []))
        return count

    def _count_functions(self) -> int:
        """
        Подсчитывает функции в проекте

        Returns:
            int: общее количество функций
        """
        count = 0
        for file_path, analyzer in self.file_analyzers.items ():
            result = analyzer.parse ()
            if result and "error" not in result:
                count += len (result.get ("functions", []))

                # Методы классов Python
                for cls in result.get ("classes", []):
                    count += len (cls.get ("methods", []))

                # Методы имплементаций Rust
                for impl in result.get ("impls", []):
                    count += len (impl.get ("methods", []))

                # Методы контрактов Solidity
                for contract in result.get ("contracts", []):
                    count += len (contract.get ("functions", []))
        return count

    def _extract_external_dependencies(self) -> List[str]:
        """
        Извлекает внешние зависимости проекта

        Returns:
            List[str]: список внешних зависимостей
        """
        external_deps = set ()
        for file_path, deps in self.dependency_graph.items ():
            for dep, is_internal in deps.items ():
                if not is_internal:
                    external_deps.add (dep)
        return sorted (list (external_deps))

    def _build_internal_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Строит граф внутренних зависимостей

        Returns:
            Dict[str, List[str]]: граф внутренних зависимостей
        """
        internal_deps = {}
        for file_path, deps in self.dependency_graph.items ():
            internal_deps[file_path] = []
            for dep, is_internal in deps.items ():
                if is_internal:
                    # Попытка сопоставить относительный импорт с реальным файлом
                    target_file = self._resolve_relative_import (file_path, dep)
                    if target_file:
                        internal_deps[file_path].append (target_file)
        return internal_deps

    def _resolve_relative_import(self, source_file: str, import_path: str) -> Optional[str]:
        """
        Пытается сопоставить относительный импорт с реальным файлом

        Args:
            source_file: исходный файл
            import_path: путь импорта

        Returns:
            Optional[str]: путь к файлу или None, если не найден
        """
        if not import_path.startswith ('.'):
            return None

        # Получаем директорию исходного файла
        source_dir = os.path.dirname (source_file)

        # Удаляем начальные точки
        path = import_path
        level = 0
        while path.startswith ('.'):
            path = path[1:]
            level += 1

        # Поднимаемся на уровень выше для каждой точки кроме первой
        for _ in range (level - 1):
            source_dir = os.path.dirname (source_dir)

        # Добавляем правильное расширение файла
        possible_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.rs', '.sol']

        # Преобразуем импорт в путь файла
        module_path = path.replace ('.', '/')

        # Проверяем различные варианты путей
        for ext in possible_extensions:
            full_path = os.path.normpath (os.path.join (source_dir, module_path + ext))
            relative_path = os.path.relpath (full_path, self.project_path)
            if relative_path in self.file_paths:
                return relative_path

            # Проверяем как пакет (директорию с __init__.py)
            if ext == '.py':
                init_path = os.path.normpath (os.path.join (source_dir, module_path, '__init__.py'))
                init_relative = os.path.relpath (init_path, self.project_path)
                if init_relative in self.file_paths:
                    return init_relative

        return None





