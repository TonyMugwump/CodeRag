// js_parser.js
// Node.js script to parse JavaScript files and output functions, classes, and imports as JSON
// Usage: node js_parser.js <file_path>

const fs = require('fs');
const esprima = require('esprima');
const path = require('path');

function log(msg) {
    const now = new Date().toISOString();
    process.stderr.write(`[${now}] ${msg}\n`);
}

function extractElements(ast, source, filePath) {
    const elements = [];
    const callEdges = [];
    function getSource(node) {
        return source.slice(node.range[0], node.range[1]);
    }
    function walk(node, parent = null, currentFunction = null) {
        if (!node) return;
        // Functions (declaration)
        if (node.type === 'FunctionDeclaration') {
            const funcName = node.id ? node.id.name : '<anonymous>';
            const element = {
                name: funcName,
                type: 'function',
                file_path: filePath,
                line_start: node.loc.start.line,
                line_end: node.loc.end.line,
                source_code: getSource(node),
                docstring: null,
                parent: parent && parent.type === 'ClassDeclaration' ? parent.id.name : null,
                children: [],
                dependencies: [],
                calls: [],
                complexity: 0
            };
            elements.push(element);
            // Walk function body, track calls
            if (node.body && node.body.body) {
                for (const stmt of node.body.body) {
                    walk(stmt, node, funcName);
                }
            }
            return;
        }
        // Function Expressions & Arrow Functions
        if (node.type === 'FunctionExpression' || node.type === 'ArrowFunctionExpression') {
            // Try to get a name if possible (variable, property, method, etc.)
            let funcName = '<anonymous>';
            if (parent && parent.type === 'VariableDeclarator' && parent.id && parent.id.name) {
                funcName = parent.id.name;
            } else if (parent && parent.type === 'AssignmentExpression' && parent.left && parent.left.type === 'Identifier') {
                funcName = parent.left.name;
            } else if (parent && parent.type === 'Property' && parent.key && parent.key.name) {
                funcName = parent.key.name;
            } else if (parent && parent.type === 'MethodDefinition' && parent.key && parent.key.name) {
                funcName = parent.key.name;
            }
            const element = {
                name: funcName,
                type: node.type === 'ArrowFunctionExpression' ? 'arrow_function' : 'function_expression',
                file_path: filePath,
                line_start: node.loc.start.line,
                line_end: node.loc.end.line,
                source_code: getSource(node),
                docstring: null,
                parent: parent && parent.type === 'ClassDeclaration' ? parent.id.name : null,
                children: [],
                dependencies: [],
                calls: [],
                complexity: 0
            };
            elements.push(element);
            // Walk function body, track calls
            if (node.body) {
                if (Array.isArray(node.body.body)) {
                    for (const stmt of node.body.body) {
                        walk(stmt, node, funcName);
                    }
                } else {
                    // Arrow function with expression body
                    walk(node.body, node, funcName);
                }
            }
            return;
        }
        // Classes
        if (node.type === 'ClassDeclaration') {
            const methods = [];
            if (node.body && node.body.body) {
                for (const method of node.body.body) {
                    if (method.type === 'MethodDefinition') {
                        methods.push(method.key.name);
                        walk(method.value, node, method.key.name);
                    }
                }
            }
            elements.push({
                name: node.id ? node.id.name : '<anonymous>',
                type: 'class',
                file_path: filePath,
                line_start: node.loc.start.line,
                line_end: node.loc.end.line,
                source_code: getSource(node),
                docstring: null,
                parent: null,
                children: methods,
                dependencies: [],
                calls: [],
                complexity: 0
            });
            return;
        }
        // Imports (ES6)
        if (node.type === 'ImportDeclaration') {
            elements.push({
                name: node.source.value,
                type: 'import',
                file_path: filePath,
                line_start: node.loc.start.line,
                line_end: node.loc.end.line,
                source_code: getSource(node),
                docstring: null,
                parent: null,
                children: [],
                dependencies: [],
                calls: [],
                complexity: 0
            });
            return;
        }
        // Calls (for call graph)
        if (node.type === 'CallExpression' && node.callee) {
            // Improved: build full callee path for member expressions (e.g., obj.method, this.method)
            let callName = null;
            if (node.callee.type === 'Identifier') callName = node.callee.name;
            else if (node.callee.type === 'MemberExpression') {
                // Build full callee path (e.g., obj.method or this.method)
                function getMemberExprName(expr) {
                    if (expr.type === 'Identifier') return expr.name;
                    if (expr.type === 'ThisExpression') return 'this';
                    if (expr.type === 'MemberExpression') {
                        return getMemberExprName(expr.object) + '.' + getMemberExprName(expr.property);
                    }
                    return '?';
                }
                callName = getMemberExprName(node.callee);
            }
            if (callName && currentFunction) {
                callEdges.push({
                    from: { name: currentFunction, file: filePath },
                    to: { name: callName, file: filePath }
                });
            }
        }
        for (const key in node) {
            if (node[key] && typeof node[key] === 'object' && key !== 'loc' && key !== 'range') {
                if (Array.isArray(node[key])) {
                    node[key].forEach(child => walk(child, node, currentFunction));
                } else {
                    walk(node[key], node, currentFunction);
                }
            }
        }
    }
    walk(ast, null, null);
    return { elements, callEdges };
}

function parseFile(filePath) {
    try {
        // Enhanced error handling: check for file existence and empty files
        const absPath = path.resolve(filePath);
        if (!fs.existsSync(absPath)) {
            log(`WARNING: File not found: ${filePath}`);
            return { elements: [], callEdges: [] };
        }
        const source = fs.readFileSync(absPath, 'utf-8');
        if (!source.trim()) {
            log(`WARNING: File is empty: ${filePath}`);
            return { elements: [], callEdges: [] };
        }
        let ast;
        try {
            // Parse with esprima, catch syntax errors
            ast = esprima.parseModule(source, { loc: true, range: true });
        } catch (e) {
            log(`ERROR parsing ${filePath}: ${e.message}`);
            return { elements: [], callEdges: [] };
        }
        const result = extractElements(ast, source, filePath);
        if ((!result.elements || result.elements.length === 0) && (!result.callEdges || result.callEdges.length === 0)) {
            log(`WARNING: No elements or call edges found in ${filePath}`);
        }
        return result;
    } catch (e) {
        log(`FATAL ERROR parsing ${filePath}: ${e.message}`);
        return { elements: [], callEdges: [] };
    }
}

function main() {
    let fileList = [];
    if (process.argv.length > 2) {
        // Accept files as CLI arguments (batch mode)
        fileList = process.argv.slice(2);
    } else {
        // Accept JSON array of files via stdin
        let input = '';
        process.stdin.setEncoding('utf-8');
        process.stdin.on('data', chunk => input += chunk);
        process.stdin.on('end', () => {
            try {
                fileList = JSON.parse(input);
                runBatch(fileList);
            } catch (e) {
                log('ERROR: Invalid input. Provide a JSON array of file paths.');
                process.exit(1);
            }
        });
        return;
    }
    runBatch(fileList);
}

function runBatch(fileList) {
    try {
        // Wrap batch in try/catch to ensure valid output on fatal errors
        log(`Batch parse started. Files: ${fileList.length}`);
        const t0 = Date.now();
        let allElements = [];
        let allCallEdges = [];
        for (const file of fileList) {
            const t1 = Date.now();
            log(`Parsing: ${file}`);
            const { elements, callEdges } = parseFile(file);
            log(`Parsed: ${file} (${elements.length} elements, ${Date.now() - t1} ms)`);
            allElements = allElements.concat(elements);
            allCallEdges = allCallEdges.concat(callEdges);
        }
        log(`Batch parse finished. Total elements: ${allElements.length}. Time: ${Date.now() - t0} ms`);
        process.stdout.write(JSON.stringify({ elements: allElements, callEdges: allCallEdges }));
    } catch (e) {
        log(`FATAL ERROR in batch: ${e.message}`);
        process.stdout.write(JSON.stringify({ elements: [], callEdges: [] }));
    }
}

main();
