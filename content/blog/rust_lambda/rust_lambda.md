---
title: Using Rust for AWS Lambdas
date: 2021-03-02
description: "Making a performant async lambda using rust : A tutorial, and my experiences along the way"
tags:
  - rust
  - web
  - AWS
---

> Discussion on [reddit](https://www.reddit.com/r/rust/comments/lwticq/creating_an_aws_lambda_with_rust/).

What we'll be working towards in this article is making a [AWS Lambda](https://aws.amazon.com/lambda/) that calls an API (an asynchronous action) and fetches a quote for us. AWS Lambdas are a good way to do this, and I'll be referring to them as just "Lambdas" in this article.

We'll use Rust to ship a single executable binary which our lambda can run when invoked. It's not quite as straightfoward as it should be form the rust side of things, mainly due to lambda rust runtime, at the time of writing. Hopefully this article helps you navigate those waters.

I really like articles with code. I've created a [repository for the code](https://github.com/RohanGautam/rust-aws-lambda) in this article here. Do refer to it for a more fine-grained code example. This aricle just explains the main ideas behind things.

### The development environment

Lambdas have this random quirk about running in an _Amazon Linux_ environment, which is a flavor of linux developed by Amazon based on RedHat. What this implies to you as a rust developer, is that your code has to target the `x86_64-unknown-linux-musl` build platform.

The official AWS Docs explain how to do this [in detail](https://aws.amazon.com/es/blogs/opensource/rust-runtime-for-aws-lambda/), but I ran into OpenSSL build issues on my M1 Mac as well as an Ubuntu instance, so I opted with using an Amazon linux 2 instance to develop it in the first place, where it built without surprises after installing some missing packages:

```bash
sudo yum install gcc
sudo yum install openssl-devel
```

### Developing the actual service

The [official example](https://aws.amazon.com/es/blogs/opensource/rust-runtime-for-aws-lambda/) is ok, but doesn't work well at all when it comes to async code. That's because the [`aws-lambda-rust-runtime`](https://github.com/awslabs/aws-lambda-rust-runtime)'s async support is pretty recent, and the [version on crates.io](https://crates.io/crates/lambda_runtime) is severly outdated.

The solution was to include the library from github and not crates.io. You can specify the repository and the commit hash in your `Cargo.toml` :

```toml
[dependencies]
lambda = { git = "https://github.com/awslabs/aws-lambda-rust-runtime/", rev = "ba696878310347f6610db819e3824be1b798fe63"
```

Now, (finally), you can make your async lambda handler:

```rust
// ...
use lambda::{handler_fn, Context};
// ...
type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
#[tokio::main]
async fn main() -> Result<(), Error> {
    // `my_handler` is an async function where you can run your tasks
    let func = handler_fn(my_handler);
    lambda::run(func).await?;
    Ok(())
}
// ...
async fn my_handler(e: CustomEvent, c: Context) -> Result<CustomOutput, Error> {
    // dummy async fn below
    let result: bool = example::async_fn().await?;
    // `CustomEvent` and `CustomOutput` are Json-serializable structs that you define
    Ok(CustomOutput {
        message: format!("Hello from lambda"),
    })
}
```

And that's it! Pretty elegant if not for the extremely convoluted way I had to find this out. Hopefully you can now `cargo build --release` sucessfully.

### Deploying it to AWS lambda

At this point, you should have a release present at `target/release/bootstrap`. Fortunately, it's quite straightfoward from now on. There are ways to upload the binary to AWS through the AWS CLI, but I found it straightfoward to use a zip file.

```bash
zip -j rust-test-lambda.zip ./target/release/bootstrap
```

I was working on another instance, so i had to [`scp`](https://www.geeksforgeeks.org/scp-command-in-linux-with-examples/) this zip file back to my machine.

Create a lambda :

{% image "./lambda-create.png", "Create lambda"%}

I'm using the new lambda console, this is what you should see after creating one

{% image "./lambda-post-create.png", "Post Create lambda"%}

Upload your zip file :

{% image "./zip.png", "upload zip file"%}

Navigate to the test tab and create a test invocation with the data payload:

{% image "./payload.png", "data payload"%}

Marvel at the (hopefully) successful response! Do note that the high execution time in the photo below is because of the quotes API invocation that I'm making in my [example code](https://github.com/RohanGautam/rust-aws-lambda).

{% image "./result.png", "data payload"%}

### Closing thoughts

Just to plug it in again, the rust code is present in [this repository](https://github.com/RohanGautam/rust-aws-lambda), refer to it for the full details and the Quote Lambda Example!

In my opinon this whole process highlights some pros and cons of the state of rust. The pros are the extremely fast, safe and performant code you can write with rust. This literally translates to monetary savings when used with serveless executors like AWS Lambda. The cons are the state of dependency maintainance for critial and popular applications like these, and it _feels_ like I should just be able to rely on the crates from cargo for my needs.
