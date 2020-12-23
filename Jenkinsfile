elifePipeline {
    node('containers-jenkins-plugin') {
        def commit
        def baseGrobidTag
        def version

        stage 'Checkout', {
            checkout scm
            commit = elifeGitRevision()
            baseGrobidTag = sh(
                script: 'bash -c "source .env && echo \\$BASE_GROBID_TAG"',
                returnStdout: true
            ).trim()
            echo "baseGrobidTag: ${baseGrobidTag}"
            assert baseGrobidTag != ''
            if (env.TAG_NAME) {
                version = env.TAG_NAME - 'v'
            } else {
                version = 'develop'
            }
        }

        stage 'Build and run tests', {
            def actions = [
                'ci-build-and-test-core': {
                    withCommitStatus({
                        sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-core"
                    }, 'ci-build-core', commit)
                    def subActions = [
                        'ci-lint': {
                            withCommitStatus({
                                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-lint"
                            }, 'ci-lint', commit)
                        },
                        'ci-pytest-not-slow': {
                            withCommitStatus({
                                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-pytest-not-slow"
                            }, 'ci-pytest-not-slow', commit)
                        },
                        'ci-pytest-slow': {
                            withCommitStatus({
                                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-pytest-slow"
                            }, 'ci-pytest-slow', commit)
                        },
                        'ci-test-setup-install': {
                            withCommitStatus({
                                sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-test-setup-install"
                            }, 'ci-test-setup-install', commit)
                        }
                    ]
                    parallel subActions
                },
                'ci-build-grobid': {
                    withCommitStatus({
                        sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-grobid"
                    }, 'ci-build-grobid', commit)
                },
                'ci-build-grobid-trainer': {
                    withCommitStatus({
                        sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-grobid-trainer"
                    }, 'ci-build-grobid-trainer', commit)
                },
                'ci-build-and-test-jupyter': {
                    withCommitStatus({
                        sh "make IMAGE_TAG=${commit} REVISION=${commit} ci-build-and-test-jupyter"
                    }, 'ci-build-and-test-jupyter', commit)
                }
            ]
            try {
                parallel actions
            } finally {
                sh "make ci-clean"
            }
        }

        elifeMainlineOnly {
            stage 'Merge to master', {
                elifeGitMoveToBranch commit, 'master'
            }

            stage 'Push unstable sciencebeam-trainer-delft image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft', commit)
                def unstable_image = image.addSuffixAndTag('_unstable', commit)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }

            stage 'Push unstable sciencebeam-trainer-delft-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-grobid', tag)
                def unstable_image = image.addSuffixAndTag('_unstable', tag)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }

            stage 'Push unstable sciencebeam-trainer-delft-trainer-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-trainer-grobid', tag)
                def unstable_image = image.addSuffixAndTag('_unstable', tag)
                unstable_image.tag('latest').push()
                unstable_image.push()
            }
        }

        elifeTagOnly { repoTag ->
            stage 'Push stable sciencebeam-trainer-delft image', {
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft', commit)
                image.tag('latest').push()
                image.tag(version).push()
            }

            stage 'Push stable sciencebeam-trainer-delft-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-grobid', tag)
                image.tag('latest').push()
                image.tag(version).push()
            }

            stage 'Push stable sciencebeam-trainer-delft-trainer-grobid image', {
                def tag = "${baseGrobidTag}-${commit}"
                def image = DockerImage.elifesciences(this, 'sciencebeam-trainer-delft-trainer-grobid', tag)
                image.tag('latest').push()
                image.tag(version).push()
            }
        }
    }
}
