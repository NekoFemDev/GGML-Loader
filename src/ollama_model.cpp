#include "ollama_model.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <vector>

namespace {

bool file_exists(const std::string & path) {
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool read_whole_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Ollama's model reference syntax (see ollama's own `types/model.ParseName`), simplified to what
// `ollama pull`/`ollama list` actually produce: [host/][namespace/]model[:tag]. Bare names default
// to host=registry.ollama.ai, namespace=library, tag=latest -- the common case for `ollama pull
// <name>` / `ollama pull <name>:<tag>` with no registry involved.
struct ollama_ref {
    std::string host      = "registry.ollama.ai";
    std::string ns        = "library";
    std::string model;
    std::string tag       = "latest";
};

ollama_ref parse_ollama_ref(const std::string & spec) {
    ollama_ref r;
    std::string rest = spec;

    // split off ":tag" from the last colon, but only if nothing after it contains a '/' (so we
    // don't misparse a host:port -- not expected from `ollama pull` output, but be safe)
    size_t colon = rest.rfind(':');
    if (colon != std::string::npos && rest.find('/', colon) == std::string::npos) {
        r.tag = rest.substr(colon + 1);
        rest = rest.substr(0, colon);
    }

    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t i = 0; i <= rest.size(); i++) {
        if (i == rest.size() || rest[i] == '/') {
            parts.push_back(rest.substr(start, i - start));
            start = i + 1;
        }
    }

    if (parts.size() == 1) {
        r.model = parts[0];
    } else if (parts.size() == 2) {
        r.ns = parts[0];
        r.model = parts[1];
    } else if (parts.size() >= 3) {
        r.host = parts[0];
        r.ns = parts[1];
        r.model = parts[2];
        for (size_t i = 3; i < parts.size(); i++) r.model += "/" + parts[i];
    }
    return r;
}

std::string models_dir() {
    const char * env = getenv("OLLAMA_MODELS");
    if (env && *env) return env;
    const char * home = getenv("HOME");
    return (home ? std::string(home) : std::string(".")) + "/.ollama/models";
}

// Finds the index just past the closing bracket matching the opening bracket at `open_pos`
// (which must point at '{' or '['), honoring quoted strings/escapes so braces inside string
// values don't confuse the depth count. Returns std::string::npos if unbalanced.
size_t find_matching_bracket(const std::string & s, size_t open_pos) {
    char open = s[open_pos];
    char close = (open == '{') ? '}' : ']';
    int depth = 0;
    bool in_str = false;
    for (size_t i = open_pos; i < s.size(); i++) {
        char c = s[i];
        if (in_str) {
            if (c == '\\') { i++; continue; }
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == open) depth++;
        else if (c == close) {
            depth--;
            if (depth == 0) return i + 1;
        }
    }
    return std::string::npos;
}

// splits a "[{...},{...},...]" array body (brackets included) into its top-level "{...}" objects
std::vector<std::string> split_top_level_objects(const std::string & arr) {
    std::vector<std::string> out;
    for (size_t i = 0; i < arr.size(); i++) {
        if (arr[i] == '{') {
            size_t end = find_matching_bracket(arr, i);
            if (end == std::string::npos) break;
            out.push_back(arr.substr(i, end - i));
            i = end - 1;
        }
    }
    return out;
}

// extracts the string value of "key":"value" from a flat JSON object fragment (no nested-object
// values for the keys we care about here, so a plain substring search is enough)
bool extract_string_field(const std::string & obj, const std::string & key, std::string & out) {
    std::string needle = "\"" + key + "\"";
    size_t kp = obj.find(needle);
    if (kp == std::string::npos) return false;
    size_t colon = obj.find(':', kp + needle.size());
    if (colon == std::string::npos) return false;
    size_t q1 = obj.find('"', colon);
    if (q1 == std::string::npos) return false;
    size_t q2 = q1 + 1;
    while (q2 < obj.size() && obj[q2] != '"') {
        if (obj[q2] == '\\') q2++;
        q2++;
    }
    if (q2 >= obj.size()) return false;
    out = obj.substr(q1 + 1, q2 - q1 - 1);
    return true;
}

// finds the GGUF model blob's digest ("sha256:...") among the manifest's "layers" entries, i.e.
// the one whose mediaType is Ollama's model-weights layer (as opposed to the system prompt,
// chat template, or license layers also listed there)
bool find_model_layer_digest(const std::string & manifest_json, std::string & digest) {
    size_t lp = manifest_json.find("\"layers\"");
    if (lp == std::string::npos) return false;
    size_t arr_start = manifest_json.find('[', lp);
    if (arr_start == std::string::npos) return false;
    size_t arr_end = find_matching_bracket(manifest_json, arr_start);
    if (arr_end == std::string::npos) return false;

    std::string arr = manifest_json.substr(arr_start, arr_end - arr_start);
    for (const std::string & obj : split_top_level_objects(arr)) {
        std::string media_type, dig;
        if (!extract_string_field(obj, "mediaType", media_type)) continue;
        if (media_type != "application/vnd.ollama.image.model") continue;
        if (!extract_string_field(obj, "digest", dig)) continue;
        digest = dig;
        return true;
    }
    return false;
}

} // namespace

std::string resolve_model_ref(const std::string & spec) {
    if (file_exists(spec)) return spec;

    ollama_ref ref = parse_ollama_ref(spec);
    std::string dir = models_dir();
    std::string manifest_path = dir + "/manifests/" + ref.host + "/" + ref.ns + "/" + ref.model + "/" + ref.tag;

    std::string manifest_json;
    if (!read_whole_file(manifest_path, manifest_json)) {
        fprintf(stderr,
            "note: '%s' is not a file and not a manifest under %s\n"
            "      (looked for %s -- if this is meant to be an Ollama model, check `ollama list`\n"
            "      and that OLLAMA_MODELS/HOME point at the right place)\n",
            spec.c_str(), dir.c_str(), manifest_path.c_str());
        return spec;
    }

    std::string digest;
    if (!find_model_layer_digest(manifest_json, digest)) {
        fprintf(stderr, "error: manifest '%s' has no model-weights layer (unexpected format)\n",
                manifest_path.c_str());
        return spec;
    }

    // manifest digests are "sha256:<hex>"; Ollama's blob files on disk use "sha256-<hex>" (colon
    // isn't a valid filename character)
    std::string blob_name = digest;
    size_t c = blob_name.find(':');
    if (c != std::string::npos) blob_name[c] = '-';
    std::string blob_path = dir + "/blobs/" + blob_name;

    if (!file_exists(blob_path)) {
        fprintf(stderr, "error: manifest '%s' points at missing blob '%s' (incomplete pull?)\n",
                manifest_path.c_str(), blob_path.c_str());
        return spec;
    }

    fprintf(stderr, "resolved Ollama model '%s' -> %s\n", spec.c_str(), blob_path.c_str());
    return blob_path;
}
