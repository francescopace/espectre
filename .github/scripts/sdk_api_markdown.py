#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Render the public Doxygen XML as one versioned registry API.md."""

from __future__ import annotations

import html
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def plain(node: ET.Element | None) -> str:
    if node is None:
        return ""
    if node.tag == "sp":
        return " "
    return (node.text or "") + "".join(plain(child) + (child.tail or "") for child in node)


def escape(text: str) -> str:
    return re.sub(r"([\\`*_\[\]])", r"\\\1", html.escape(text, quote=False))


def code(text: str) -> str:
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    return f"{fence} {text.strip()} {fence}" if "`" in text else f"{fence}{text.strip()}{fence}"


def code_block(text: str) -> str:
    fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    return f"\n\n{fence}cpp\n{text.strip()}\n{fence}\n\n"


def template_declaration(node: ET.Element) -> str:
    params = []
    for param in node.findall("templateparamlist/param"):
        declaration = (plain(param.find("type")) + " " + param.findtext("declname", "")).strip()
        default = plain(param.find("defval"))
        params.append(declaration + (" = " + default if default else ""))
    return "template <" + ", ".join(params) + ">\n" if params else ""


class MarkdownReference:
    def __init__(self, xml_directory: Path):
        kinds = {"page": 0, "group": 0, "namespace": 1, "class": 2, "struct": 2, "union": 2, "file": 3}
        index = ET.parse(xml_directory / "index.xml").getroot()
        self.compounds = []
        for entry in index:
            if entry.get("kind") not in kinds or entry.findtext("name", "").endswith(".dox"):
                continue
            compound = ET.parse(xml_directory / f"{entry.attrib['refid']}.xml").find("compounddef")
            if compound is None:
                raise ValueError(f"Missing Doxygen compound: {entry.attrib['refid']}")
            self.compounds.append(compound)
        self.compounds.sort(key=lambda node: (
            kinds[node.attrib["kind"]], node.attrib["id"] != "indexpage", node.findtext("compoundname", ""),
        ))
        self.anchors = {
            node.attrib["id"] for compound in self.compounds for node in compound.iter()
            if "id" in node.attrib and node.tag in {
                "compounddef", "memberdef", "enumvalue", "sect1", "sect2", "sect3", "sect4", "anchor",
            }
        }
        self.emitted: set[str] = set()

    def anchor(self, node: ET.Element) -> str:
        refid = node.attrib["id"]
        if refid in self.emitted:
            return ""
        self.emitted.add(refid)
        return f'\n\n<a id="{html.escape(refid, quote=True)}"></a>\n\n'

    def link(self, label: str, refid: str) -> str:
        return f"[{label}](#{refid})" if refid in self.anchors else label

    def contents(self, node: ET.Element) -> str:
        def text(value: str | None) -> str:
            return escape(re.sub(r"\s+", " ", value or ""))
        return text(node.text) + "".join(self.render(child) + text(child.tail) for child in node)

    def render(self, node: ET.Element | None) -> str:
        if node is None:
            return ""
        tag = node.tag
        if tag == "programlisting":
            return code_block("\n".join(plain(line) for line in node.findall("codeline")))
        if tag == "computeroutput":
            label = code(plain(node))
            refs = node.findall("ref")
            return self.link(label, refs[0].attrib["refid"]) if len(refs) == 1 else label
        if tag == "anchor":
            return self.anchor(node)
        if re.fullmatch(r"sect[1-4]", tag):
            heading = "#" * (int(tag[-1]) + 2)
            return self.anchor(node) + f"{heading} {self.render(node.find('title')).strip()}\n\n" + "".join(
                self.render(child) for child in node if child.tag != "title"
            )
        if tag == "parameterlist":
            label = {"param": "Parameters", "retval": "Return values", "exception": "Exceptions"}.get(
                node.get("kind"), "Parameters",
            )
            items = []
            for item in node.findall("parameteritem"):
                names = ", ".join(code(plain(name)) for name in item.findall("parameternamelist/parametername"))
                description = self.render(item.find("parameterdescription")).strip()
                items.append(f"- {names}: {description.replace(chr(10), chr(10) + '  ')}")
            return f"\n\n**{label}**\n\n" + "\n".join(items) + "\n\n"
        if tag in {"itemizedlist", "orderedlist"}:
            items = []
            for i, item in enumerate(node.findall("listitem"), 1):
                prefix = f"{i}. " if tag == "orderedlist" else "- "
                body = self.contents(item).strip().replace("\n", "\n" + " " * len(prefix))
                items.append(prefix + body)
            return "\n\n" + "\n\n".join(items) + "\n\n"
        if tag == "table":
            rows = ["| " + " | ".join(
                self.contents(cell).strip().replace("|", "\\|").replace("\n", "<br>")
                for cell in row.findall("entry")
            ) + " |" for row in node.findall("row")]
            rows.insert(1, "| " + " | ".join("---" for _ in node.find("row")) + " |")
            return "\n\n" + "\n".join(rows) + "\n\n"
        body = self.contents(node)
        if tag == "ref":
            return self.link(body, node.attrib["refid"])
        if tag == "ulink":
            return f"[{body}](<{node.attrib['url']}>)"
        if tag in {"bold", "emphasis"}:
            marker = "**" if tag == "bold" else "*"
            return marker + body.strip() + marker
        if tag == "simplesect":
            label = {"return": "Returns", "see": "See also"}.get(node.get("kind"), node.get("kind", "Note").capitalize())
            return f"\n\n**{label}:** {body.strip()}\n\n"
        if tag in {"para", "listitem", "xrefsect", "varlistentry"}:
            return "\n\n" + body.strip() + "\n\n"
        if tag == "xreftitle":
            return f"**{body.strip()}:** "
        if tag == "linebreak":
            return "  \n"
        if tag in {"briefdescription", "detaileddescription", "inbodydescription", "title",
                   "parameterdescription", "xrefdescription", "variablelist", "term"}:
            return body
        raise ValueError(f"Unsupported Doxygen documentation element: {tag}")

    def description(self, node: ET.Element) -> str:
        return "".join(self.render(node.find(tag)) for tag in (
            "briefdescription", "detaileddescription", "inbodydescription",
        ))

    def member(self, node: ET.Element) -> str:
        kind = node.attrib["kind"]
        name = node.findtext("name", "")
        signature = plain(node.find("definition")) + plain(node.find("argsstring"))
        if kind == "define":
            params = [param.findtext("defname", "") for param in node.findall("param")]
            signature = "#define " + name + ("(" + ", ".join(params) + ")" if params else "")
        elif kind == "enum":
            signature = "enum " + ("class " if node.get("strong") == "yes" else "") + name
            if plain(node.find("type")):
                signature += " : " + plain(node.find("type"))
        initializer = plain(node.find("initializer"))
        if initializer:
            signature += " " + initializer
        for keyword in ("explicit", "constexpr"):
            if node.get(keyword) == "yes" and not re.search(rf"\b{keyword}\b", signature):
                signature = keyword + " " + signature
        signature = template_declaration(node) + signature
        result = self.anchor(node) + f"### {code(name)}\n" + code_block(signature) + self.description(node)
        for value in node.findall("enumvalue"):
            declaration = value.findtext("name", "") + " " + plain(value.find("initializer"))
            result += self.anchor(value) + code(declaration) + "\n\n" + self.description(value)
        return result

    def document(self, version: str, commit: str) -> str:
        result = f"# ESPectre SDK C++ API\n\nVersion: {code(version)}\n\nSource commit: {code(commit)}\n\n"
        result += "[SDK guide](README.md)\n\n## Contents\n\n"
        for compound in self.compounds:
            title = compound.findtext("title") or compound.findtext("compoundname", "")
            result += f"- {self.link(escape(title), compound.attrib['id'])}\n"
        for compound in self.compounds:
            title = compound.findtext("title") or compound.findtext("compoundname", "")
            result += self.anchor(compound) + f"## {escape(title)}\n\n"
            for include in compound.findall("includes"):
                result += code_block(f'#include <{plain(include)}>')
            if compound.get("kind") in {"class", "struct", "union"}:
                declaration = template_declaration(compound) + f"{compound.attrib['kind']} {title}"
                bases = [node.get("prot", "public") + " " + plain(node) for node in compound.findall("basecompoundref")]
                if bases:
                    declaration += " : " + ", ".join(bases)
                result += code_block(declaration)
            result += self.description(compound)
            for tag in ("basecompoundref", "innerclass", "innernamespace"):
                for node in compound.findall(tag):
                    result += self.link(code(plain(node)), node.get("refid", "")) + "\n\n"
            for section in compound.findall("sectiondef"):
                for member in section.findall("memberdef"):
                    if member.attrib["id"] not in self.emitted:
                        result += self.member(member)
                    else:
                        result += "- " + self.link(code(member.findtext("name", "")), member.attrib["id"]) + "\n"
                result += "\n"
        missing = self.anchors - self.emitted
        if missing:
            raise ValueError(f"Unrendered Doxygen anchors: {sorted(missing)}")
        return result.rstrip() + "\n"


def generate_registry_api(bundle_root: Path, destination: Path, version: str, commit: str) -> None:
    from build_sdk_package import stamp_doxyfile_project_number, stamp_sdk_version_header
    from generate_sdk_api import prune_private_members, run_doxygen, stamp_doxyfile_output_directory

    # Keep bundle identity unchanged while documenting the registry's exact macros.
    with tempfile.TemporaryDirectory(prefix="espectre-registry-api-") as tmp:
        source_root = Path(tmp)
        cpp_root = source_root / "src" / "cpp"
        shutil.copytree(bundle_root / "src" / "cpp", cpp_root)
        stamp_sdk_version_header(cpp_root / "runtime" / "espectre_sdk_version.h", version)
        doxyfile = cpp_root / "Doxyfile"
        stamp_doxyfile_project_number(doxyfile, version)
        stamp_doxyfile_output_directory(doxyfile, source_root / "output")
        run_doxygen(doxyfile, source_root)
        xml_directory = source_root / "output" / "xml"
        prune_private_members(xml_directory)
        reference = MarkdownReference(xml_directory)
        markdown = reference.document(version, commit)
        (destination / "API.md").write_text(markdown, encoding="utf-8")
        readme = destination / "README.md"
        readme.write_text(
            readme.read_text(encoding="utf-8").rstrip()
            + "\n\n## C++ API reference\n\n"
            "[API.md](API.md) contains the complete C++ reference and integration contracts "
            "for this component's version and source commit.\n",
            encoding="utf-8",
        )
