const path = require("path");
const eleventyImage = require("@11ty/eleventy-img");

function relativeToInputPath(inputPath, relativeFilePath) {
	let split = inputPath.split("/");
	split.pop();

	return path.resolve(split.join(path.sep), relativeFilePath);
}

function isFullUrl(url) {
	try {
		new URL(url);
		return true;
	} catch (e) {
		return false;
	}
}

module.exports = function (eleventyConfig) {
	// Eleventy Image shortcode
	// https://www.11ty.dev/docs/plugins/image/
	eleventyConfig.addAsyncShortcode(
		"image",
		async function imageShortcode(
			src,
			alt,
			figure = false,
			widths = [400, 800, 1200],
			sizes = "100vw",
		) {
			// Full list of formats here: https://www.11ty.dev/docs/plugins/image/#output-formats
			// Warning: Avif can be resource-intensive so take care!
			let formats = ["avif", "webp"];

			let input;
			if (isFullUrl(src)) {
				input = src;
			} else {
				input = relativeToInputPath(this.page.inputPath, src);
			}
			// if a gif provided, output that
			if (input.endsWith(".gif")) {
				formats = ["gif"];
			}

			let metadata = await eleventyImage(input, {
				widths: widths,
				formats,
				sharpOptions: {
					animated: true,
					fit: "contain",
				},
				outputDir: path.join(eleventyConfig.dir.output, "img"), // Advanced usage note: `eleventyConfig.dir` works here because we’re using addPlugin.
			});

			// TODO loading=eager and fetchpriority=high
			let imageAttributes = {
				alt,
				sizes,
				loading: "lazy",
				decoding: "async",
			};

			let html = eleventyImage.generateHTML(metadata, imageAttributes);
			if (figure) {
				// add alt text as figure caption
				return `<figure>${html}<figcaption>${alt}</figcaption></figure>`;
			} else {
				return html;
			}
		},
	);
};
